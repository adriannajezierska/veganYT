import os
import re
import json
import math
import logging
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = lambda: None

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore


# Initialize environment and logging
load_dotenv()
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntelligentAnalyzer")


@dataclass
class ExtractionConfig:
	min_topics: int = 2
	max_topics: int = 20
	tokens_per_topic: int = 750  # here treated as words-per-topic approximation
	keywords_per_topic_single: int = 15
	chunk_size: int = 15000  # words
	similarity_threshold: float = 0.6
	max_keywords_final: int = 15


class TopicExtractor:
	"""Extract topics from text and merge them lexically without embeddings."""

	def __init__(self, client: Any, config: Optional[Dict[str, Any]] = None) -> None:
		self.client = client
		self.config = self._normalize_config(config)
		self.progress = {'completed': [], 'results': {}}
		self.final_topics: List[Dict[str, Any]] = []
		self.documents: List[str] = []
		self.topic_word_dist: pd.DataFrame = pd.DataFrame()

	def _normalize_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		default = ExtractionConfig()
		cfg = {
			'min_topics': default.min_topics,
			'max_topics': default.max_topics,
			'tokens_per_topic': default.tokens_per_topic,
			'keywords_per_topic_single': default.keywords_per_topic_single,
			'chunk_size': default.chunk_size,
			'similarity_threshold': default.similarity_threshold,
			'max_keywords_final': default.max_keywords_final,
		}
		if config:
			cfg.update(config)
		return cfg

	# ---------- Counting and chunking ----------
	def count_tokens(self, text: str) -> int:
		# Using words as approximation
		return len(text.split())

	def estimate_topic_count(self, text: str) -> Tuple[int, int, int]:
		token_count = self.count_tokens(text)
		topics_from_tokens = max(
			self.config['min_topics'],
			min(self.config['max_topics'], token_count // self.config['tokens_per_topic'])
		)
		min_topics = max(self.config['min_topics'], topics_from_tokens - 2)
		max_topics = min(self.config['max_topics'], topics_from_tokens + 3)
		logger.info(f"Estimated topic range: {min_topics}-{max_topics} (words: {token_count})")
		return min_topics, max_topics, token_count

	def _chunk_by_words(self, text: str, max_words: int = 6000, overlap_words: int = 200) -> List[str]:
		words = text.split()
		chunks: List[str] = []
		start = 0
		while start < len(words):
			end = min(start + max_words, len(words))
			chunks.append(" ".join(words[start:end]))
			if end == len(words):
				break
			start = max(0, end - overlap_words)
		return chunks

	# ---------- Robust JSON parsing ----------
	def _strip_code_fences(self, content: str) -> str:
		content = content.strip()
		# remove ```json ... ``` fences if present
		if content.startswith("```"):
			content = re.sub(r"^```[a-zA-Z]*\n|```$", "", content, flags=re.MULTILINE).strip()
		return content

	def _safe_json(self, text: str) -> Dict[str, Any]:
		try:
			return json.loads(text)
		except Exception:
			m = re.search(r"\{[\s\S]*\}", text)
			if m:
				return json.loads(m.group(0))
			raise

	# ---------- Lexical similarity (no embeddings) ----------
	def _tokenize(self, text: str) -> List[str]:
		return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t]

	def _char_ngrams(self, text: str, n: int = 3) -> set:
		s = f" {text.lower()} "
		if len(s) < n:
			return {s}
		return {s[i:i + n] for i in range(len(s) - n + 1)}

	def _name_similarity(self, a: str, b: str) -> float:
		wa, wb = set(self._tokenize(a)), set(self._tokenize(b))
		jacc = len(wa & wb) / (len(wa | wb) + 1e-8) if (wa or wb) else 0.0
		ga, gb = self._char_ngrams(a), self._char_ngrams(b)
		dice = (2 * len(ga & gb)) / (len(ga) + len(gb) + 1e-8) if (ga and gb) else 0.0
		length_ratio = min(len(a), len(b)) / (max(len(a), len(b)) + 1e-8)
		return 0.5 * jacc + 0.35 * dice + 0.15 * length_ratio

	def _weighted_jaccard_keywords(self, kw_a: Dict[str, float], kw_b: Dict[str, float], idf: Optional[Dict[str, float]] = None) -> float:
		idf = idf or {}
		keys = set(kw_a) | set(kw_b)
		if not keys:
			return 0.0
		numerator = 0.0
		denominator = 0.0
		for k in keys:
			wa = float(kw_a.get(k, 0.0))
			wb = float(kw_b.get(k, 0.0))
			w = float(idf.get(k, 1.0))
			numerator += w * min(wa, wb)
			denominator += w * max(wa, wb)
		return numerator / (denominator + 1e-8)

	def _doc_overlap_similarity(self, a_docs: List[str], b_docs: List[str]) -> float:
		if not a_docs or not b_docs:
			return 0.0
		a, b = set(a_docs), set(b_docs)
		return len(a & b) / (len(a | b) + 1e-8)

	def _compute_idf_over_topics(self, topic_instances: List[Dict[str, Any]]) -> Dict[str, float]:
		document_frequency: Dict[str, int] = defaultdict(int)
		doc_set_by_word: Dict[str, set] = defaultdict(set)
		all_docs: set = set()
		for topic in topic_instances:
			doc_id = topic.get('doc_id', f"doc_{id(topic)}")
			all_docs.add(doc_id)
			for word in topic.get('keywords', {}).keys():
				doc_set_by_word[word].add(doc_id)
		for word, doc_ids in doc_set_by_word.items():
			document_frequency[word] = len(doc_ids)
		N = max(1, len(all_docs))
		return {word: math.log(1 + N / (1 + df_w)) for word, df_w in document_frequency.items()}

	def _topic_similarity_lexical(self, topic_a: Dict[str, Any], topic_b: Dict[str, Any], idf: Optional[Dict[str, float]] = None) -> float:
		name_similarity = self._name_similarity(topic_a.get('name', ''), topic_b.get('name', ''))
		keyword_similarity = self._weighted_jaccard_keywords(topic_a.get('keywords', {}), topic_b.get('keywords', {}), idf=idf)
		doc_similarity = self._doc_overlap_similarity(
			[topic_a.get('doc_id')] if topic_a.get('doc_id') else [],
			[topic_b.get('doc_id')] if topic_b.get('doc_id') else []
		)
		return 0.55 * name_similarity + 0.40 * keyword_similarity + 0.05 * doc_similarity

	# ---------- Core extraction ----------
	def extract_topics_intelligently(self, transcript: str, filename: str) -> Optional[Dict[str, Any]]:
		if filename in self.progress['completed']:
			return self.progress['results'].get(filename)

		min_topics, max_topics, token_count = self.estimate_topic_count(transcript)
		is_long = token_count > self.config['chunk_size']

		try:
			if is_long:
				chunks = self._chunk_by_words(
					transcript,
					max_words=min(self.config['chunk_size'], 8000),
					overlap_words=200
				)
				chunk_topics: List[Dict[str, Any]] = []
				for idx, chunk in enumerate(chunks):
					prompt = (
						f"Extract between {min_topics} and {max_topics} topics (name 2-5 words) from this chunk.\n"
						f"Return JSON strictly as: {{\"topics\":[{{\"name\": \"...\", \"keywords\": {{\"term\": score}}}}]}}\n"
						f"Chunk {idx + 1}/{len(chunks)}:\n{chunk}"
					)
					resp = self.client.chat.completions.create(
						model="gpt-4o-mini",
						messages=[
							{"role": "system", "content": "You are an expert in topic modeling and content analysis."},
							{"role": "user", "content": prompt}
						],
						temperature=0,
						max_tokens=1200,
						seed=1,
					)
					content = self._strip_code_fences(resp.choices[0].message.content)
					data = self._safe_json(content)
					for topic in data.get('topics', []):
						topic['doc_id'] = filename
						topic['chunk_id'] = idx
					chunk_topics.extend(data.get('topics', []))

				# Whole-transcript anchors
				doc_prompt = (
					f"Extract between {min_topics} and {max_topics} MAIN topics for the WHOLE transcript.\n"
					f"Return JSON strictly as: {{\"topics\":[{{\"name\": \"...\", \"keywords\": {{\"term\": score}}}}]}}\n"
					f"Transcript:\n{transcript}"
				)
				doc_resp = self.client.chat.completions.create(
					model="gpt-4o-mini",
					messages=[
						{"role": "system", "content": "You are an expert in topic modeling and content analysis."},
						{"role": "user", "content": doc_prompt}
					],
					temperature=0,
					max_tokens=1200,
					seed=1,
				)
				content_doc = self._strip_code_fences(doc_resp.choices[0].message.content)
				doc_data = self._safe_json(content_doc)
				for topic in doc_data.get('topics', []):
					topic['doc_id'] = filename
					topic['chunk_id'] = None

				result = {
					'topics': chunk_topics,
					'doc_topics': doc_data.get('topics', []),
					'text_length': len(transcript),
					'topic_count': len(chunk_topics),
					'timestamp': datetime.now().isoformat(),
				}
			else:
				prompt = (
					f"Extract between {min_topics} and {max_topics} topics (name 2-5 words).\n"
					f"Return JSON strictly as: {{\"topics\":[{{\"name\": \"...\", \"keywords\": {{\"term\": score}}}}]}}\n"
					f"Transcript:\n{transcript}"
				)
				resp = self.client.chat.completions.create(
					model="gpt-4o-mini",
					messages=[
						{"role": "system", "content": "You are an expert in topic modeling and content analysis."},
						{"role": "user", "content": prompt}
					],
					temperature=0,
					max_tokens=1500,
					seed=1,
				)
				content = self._strip_code_fences(resp.choices[0].message.content)
				data = self._safe_json(content)
				for topic in data.get('topics', []):
					topic['doc_id'] = filename
					topic['chunk_id'] = None

				result = {
					'topics': data.get('topics', []),
					'doc_topics': data.get('topics', []),
					'text_length': len(transcript),
					'topic_count': len(data.get('topics', [])),
					'timestamp': datetime.now().isoformat(),
				}

			self.progress['results'][filename] = result
			self.progress['completed'].append(filename)
			logger.info(f"✅ {filename}: topics extracted (chunked={is_long})")
			return result

		except Exception as exc:
			logger.error(f"❌ {filename}: {str(exc)}")
			return None

	# ---------- Merging without embeddings ----------
	def _merge_and_anchor_topics_lexical(self, topic_instances: List[Dict[str, Any]], summary_topics: Optional[List[Dict[str, Any]]] = None, base_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
		idf = self._compute_idf_over_topics(topic_instances)
		base_threshold = base_threshold or self.config.get('similarity_threshold', 0.6)
		anchor_threshold = max(0.45, base_threshold - 0.1)

		clusters: List[Dict[str, Any]] = []
		if summary_topics:
			for st in summary_topics:
				clusters.append({'anchor': st, 'members': [], 'name_variations': set([st['name']])})

		def add_member(cluster: Dict[str, Any], topic: Dict[str, Any]) -> None:
			cluster['members'].append(topic)
			cluster['name_variations'].add(topic['name'])

		for topic in topic_instances:
			best_index, best_similarity = -1, 0.0
			for i, cluster in enumerate(clusters):
				reference = cluster.get('anchor') or (cluster['members'][0] if cluster['members'] else None)
				if reference is None:
					continue
				similarity = self._topic_similarity_lexical(topic, reference, idf=idf)
				if similarity > best_similarity:
					best_index, best_similarity = i, similarity
			threshold = anchor_threshold if (best_index >= 0 and clusters[best_index].get('anchor')) else base_threshold
			if best_index >= 0 and best_similarity >= threshold:
				add_member(clusters[best_index], topic)
			else:
				clusters.append({'anchor': topic, 'members': [topic], 'name_variations': set([topic['name']])})

		# Second pass to merge highly similar clusters
		used = [False] * len(clusters)
		for i in range(len(clusters)):
			if used[i]:
				continue
			for j in range(i + 1, len(clusters)):
				if used[j]:
					continue
				ci, cj = clusters[i], clusters[j]
				ref_i = ci.get('anchor') or (ci['members'][0] if ci['members'] else None)
				ref_j = cj.get('anchor') or (cj['members'][0] if cj['members'] else None)
				if not ref_i or not ref_j:
					continue
				sim = self._topic_similarity_lexical(ref_i, ref_j, idf=idf)
				if sim >= base_threshold + 0.1:
					ci['members'].extend(cj['members'])
					ci['name_variations'].update(cj['name_variations'])
					used[j] = True
		clusters = [c for idx, c in enumerate(clusters) if not used[idx]]

		final_topics: List[Dict[str, Any]] = []
		for cluster in clusters:
			keywords_sum: Dict[str, float] = defaultdict(float)
			documents: set = set()
			for member in cluster['members']:
				for word, score in member.get('keywords', {}).items():
					keywords_sum[word] += float(score)
				doc_id = member.get('doc_id')
				if doc_id:
					documents.add(doc_id)
			top_keywords = dict(sorted(keywords_sum.items(), key=lambda kv: kv[1], reverse=True)[: self.config.get('max_keywords_final', 15)])
			max_value = max(top_keywords.values()) if top_keywords else 1.0
			normalized_keywords = {k: v / max_value for k, v in top_keywords.items()}
			name = (cluster.get('anchor') or {}).get('name') or max(cluster['name_variations'], key=len)
			final_topics.append({
				'name': name,
				'keywords': normalized_keywords,
				'document_count': len(documents),
				'name_variations': sorted(cluster['name_variations']),
				'member_count': len(cluster['members'])
			})
		return final_topics

	# ---------- Distributions and quality ----------
	def create_document_topic_distribution(self, topic_instances: List[Dict[str, Any]], merged_topics: List[Dict[str, Any]]) -> pd.DataFrame:
		if not merged_topics:
			return pd.DataFrame()
		idf = self._compute_idf_over_topics(topic_instances)
		document_ids = sorted({t.get('doc_id') for t in topic_instances if t.get('doc_id')})
		if not document_ids:
			return pd.DataFrame()
		index_of_doc = {doc_id: i for i, doc_id in enumerate(document_ids)}
		merged_names = [t['name'] for t in merged_topics]
		matrix = np.zeros((len(document_ids), len(merged_topics)), dtype=np.float32)

		for instance in topic_instances:
			doc_id = instance.get('doc_id')
			if not doc_id:
				continue
			best_index, best_similarity = -1, 0.0
			for i, merged in enumerate(merged_topics):
				sim = self._topic_similarity_lexical(
					instance,
					{'name': merged['name'], 'keywords': merged.get('keywords', {})},
					idf=idf
				)
				if sim > best_similarity:
					best_index, best_similarity = i, sim
			if best_index >= 0:
				weight = float(sum(instance.get('keywords', {}).values())) or 1.0
				matrix[index_of_doc[doc_id], best_index] += weight

		row_sums = matrix.sum(axis=1, keepdims=True) + 1e-8
		matrix = matrix / row_sums
		return pd.DataFrame(matrix, index=document_ids, columns=merged_names)

	def create_topic_word_distribution(self) -> pd.DataFrame:
		if not getattr(self, 'final_topics', None):
			logger.warning("No final topics available for topic-word distribution")
			self.topic_word_dist = pd.DataFrame()
			return self.topic_word_dist

		all_words: set = set()
		for topic in self.final_topics:
			all_words.update(topic.get('keywords', {}).keys())
		if not all_words:
			logger.warning("No keywords found in topics")
			self.topic_word_dist = pd.DataFrame()
			return self.topic_word_dist

		vocabulary = sorted(all_words)
		n_topics = len(self.final_topics)
		n_words = len(vocabulary)
		topic_word_matrix = np.zeros((n_topics, n_words), dtype=np.float32)
		word_index = {word: i for i, word in enumerate(vocabulary)}

		for topic_idx, topic in enumerate(self.final_topics):
			for word, score in topic.get('keywords', {}).items():
				if word in word_index:
					topic_word_matrix[topic_idx, word_index[word]] = float(score)

		row_sums = topic_word_matrix.sum(axis=1, keepdims=True)
		row_sums[row_sums == 0] = 1.0
		topic_word_matrix = topic_word_matrix / row_sums

		topic_names = [t['name'] for t in self.final_topics]
		self.topic_word_dist = pd.DataFrame(topic_word_matrix, index=topic_names, columns=vocabulary)
		return self.topic_word_dist

	def analyze_topic_quality(self) -> pd.DataFrame:
		if not getattr(self, 'final_topics', None):
			logger.warning("No final topics available for quality analysis")
			return pd.DataFrame()

		quality_rows: List[Dict[str, Any]] = []
		for topic in self.final_topics:
			keywords = topic.get('keywords', {})
			name_variations = topic.get('name_variations', [])
			metrics = {
				'topic_name': topic.get('name', ''),
				'document_coverage': (topic.get('document_count', 0) / len(self.documents)) if self.documents else 0.0,
				'keyword_diversity': len(keywords),
				'avg_keyword_weight': float(np.mean(list(keywords.values()))) if keywords else 0.0,
				'name_consistency': (1.0 / len(name_variations)) if name_variations else 1.0,
				'specificity': 0.0,
			}
			metrics['quality_score'] = (
				metrics['document_coverage'] * 0.3 +
				min(metrics['keyword_diversity'] / 20.0, 1.0) * 0.2 +
				metrics['avg_keyword_weight'] * 0.2 +
				metrics['name_consistency'] * 0.15 +
				metrics['specificity'] * 0.15
			)
			quality_rows.append(metrics)
		return pd.DataFrame(quality_rows)


class ComprehensiveAnalysisPipeline:
	"""Comprehensive analysis pipeline using TopicExtractor (no embeddings)."""

	def __init__(self, api_key: Optional[str], transcript_folder: str, output_folder: str, test_mode: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
		self.api_key = api_key or os.getenv('OPENAI_API_KEY')
		if OpenAI is None:
			raise RuntimeError("openai package not available. Please install 'openai'.")
		self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
		self.transcript_folder = transcript_folder
		self.output_folder = os.path.join(output_folder, 'test_output') if test_mode else output_folder
		self.test_mode = test_mode
		self.config = config or {}
		os.makedirs(self.output_folder, exist_ok=True)

	def _convert_numpy(self, obj: Any) -> Any:
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, dict):
			return {k: self._convert_numpy(v) for k, v in obj.items()}
		elif isinstance(obj, list):
			return [self._convert_numpy(v) for v in obj]
		return obj

	def run_analysis(self, test_limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
		logger.info("=" * 60)
		logger.info(f"Starting {'TEST MODE' if self.test_mode else 'FULL'} Topic Extraction and Modeling")
		logger.info("=" * 60)

		extractor = TopicExtractor(client=self.client, config=self.config.get('extraction', self.config))

		extraction_results: Dict[str, Dict[str, Any]] = {}
		files_processed = 0
		all_files = sorted([f for f in os.listdir(self.transcript_folder) if os.path.isfile(os.path.join(self.transcript_folder, f))])
		for filename in all_files:
			if test_limit and files_processed >= test_limit:
				break
			file_path = os.path.join(self.transcript_folder, filename)
			try:
				with open(file_path, 'r', encoding='utf-8') as fh:
					transcript = fh.read()
			except Exception as exc:
				logger.warning(f"Skipping {filename}: {exc}")
				continue
			result = extractor.extract_topics_intelligently(transcript, filename)
			if result:
				extraction_results[filename] = result
			files_processed += 1

		# Persist raw extraction results
		extraction_file = os.path.join(self.output_folder, 'extraction_results.json')
		with open(extraction_file, 'w', encoding='utf-8') as fh:
			json.dump(self._convert_numpy(extraction_results), fh, ensure_ascii=False, indent=2)

		# Prepare instances and anchors
		topic_instances: List[Dict[str, Any]] = []
		summary_topic_instances: List[Dict[str, Any]] = []
		for doc_id, res in extraction_results.items():
			for t in res.get('topics', []):
				t.setdefault('doc_id', doc_id)
				topic_instances.append(t)
			for st in res.get('doc_topics', []):
				st.setdefault('doc_id', doc_id)
				summary_topic_instances.append(st)

		# Set documents for quality analysis
		extractor.documents = list(extraction_results.keys())

		# Merge topics
		try:
			extractor.final_topics = extractor._merge_and_anchor_topics_lexical(
				topic_instances=topic_instances,
				summary_topics=summary_topic_instances,
				base_threshold=extractor.config.get('similarity_threshold', 0.6)
			)
			doc_topic_dist = extractor.create_document_topic_distribution(topic_instances, extractor.final_topics)
			topic_word_dist = extractor.create_topic_word_distribution()
			quality_df = extractor.analyze_topic_quality()
		except Exception as exc:
			logger.error(f"Error in topic modeling: {exc}")
			extractor.final_topics = []
			doc_topic_dist = pd.DataFrame()
			topic_word_dist = pd.DataFrame()
			quality_df = pd.DataFrame()

		# Persist core outputs
		if not doc_topic_dist.empty:
			doc_topic_csv = os.path.join(self.output_folder, 'doc_topic_distribution.csv')
			doc_topic_dist.to_csv(doc_topic_csv)
		if not topic_word_dist.empty:
			topic_word_csv = os.path.join(self.output_folder, 'topic_word_distribution.csv')
			topic_word_dist.to_csv(topic_word_csv)
		if not quality_df.empty:
			quality_csv = os.path.join(self.output_folder, 'topic_quality_analysis.csv')
			quality_df.to_csv(quality_csv, index=False)

		topics_file = os.path.join(self.output_folder, 'final_topics.json')
		with open(topics_file, 'w', encoding='utf-8') as fh:
			json.dump(self._convert_numpy(extractor.final_topics), fh, ensure_ascii=False, indent=2)

		# Alignment table for explainability
		try:
			alignment_df = self.build_merge_alignment_table(topic_instances, summary_topic_instances, extractor)
			alignment_path = os.path.join(self.output_folder, 'merge_alignment.csv')
			alignment_df.to_csv(alignment_path, index=False)
		except Exception as exc:
			logger.warning(f"Could not build alignment table: {exc}")

		# Report
		report = self.generate_comprehensive_report(extractor, extractor.final_topics, doc_topic_dist, topic_word_dist, quality_df, topic_instances)
		report_file = os.path.join(self.output_folder, 'analysis_report.json')
		with open(report_file, 'w', encoding='utf-8') as fh:
			json.dump(self._convert_numpy(report), fh, ensure_ascii=False, indent=2)

		# Console summary
		self.print_summary(report)

		# Excel export
		self.save_results_to_excel(doc_topic_dist, topic_word_dist, quality_df, extractor.final_topics, report)

		return doc_topic_dist, topic_word_dist, extractor.final_topics, report

	def build_merge_alignment_table(self, chunk_topics: List[Dict[str, Any]], summary_topics: List[Dict[str, Any]], extractor: TopicExtractor, sim_cutoff: float = 0.5) -> pd.DataFrame:
		idf = extractor._compute_idf_over_topics(chunk_topics + summary_topics)
		rows: List[Dict[str, Any]] = []
		for st in summary_topics:
			for ct in chunk_topics:
				sim = extractor._topic_similarity_lexical(ct, st, idf=idf)
				if sim >= sim_cutoff:
					rows.append({
						'summary_topic': st.get('name', ''),
						'chunk_topic': ct.get('name', ''),
						'chunk_doc': ct.get('doc_id'),
						'chunk_id': ct.get('chunk_id'),
						'similarity': round(float(sim), 4),
					})
		return pd.DataFrame(rows).sort_values(['summary_topic', 'similarity'], ascending=[True, False]) if rows else pd.DataFrame()

	def generate_comprehensive_report(self, extractor: TopicExtractor, final_topics: List[Dict[str, Any]], doc_topic_dist: pd.DataFrame, topic_word_dist: pd.DataFrame, quality_df: pd.DataFrame, topic_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
		document_count = len(extractor.documents)
		extracted_per_doc = []
		per_doc_counts: Dict[str, int] = defaultdict(int)
		for ti in topic_instances:
			d = ti.get('doc_id')
			if d:
				per_doc_counts[d] += 1
		extracted_per_doc = list(per_doc_counts.values())
		if not extracted_per_doc:
			extracted_per_doc = [0]

		vocab_size = len(topic_word_dist.columns) if not topic_word_dist.empty else 0

		summary = {
			'total_documents': document_count,
			'total_extracted_topics': int(sum(extracted_per_doc)),
			'final_merged_topics': int(len(final_topics)),
			'vocabulary_size': int(vocab_size),
		}

		stats = {
			'avg_topics_per_doc': float(np.mean(extracted_per_doc)),
			'std_topics_per_doc': float(np.std(extracted_per_doc)),
			'min_topics_per_doc': int(np.min(extracted_per_doc)),
			'max_topics_per_doc': int(np.max(extracted_per_doc)),
		}

		topics_sorted = sorted(final_topics, key=lambda t: t.get('document_count', 0), reverse=True)
		quality_map = {row['topic_name']: row for _, row in (quality_df.iterrows() if not quality_df.empty else [])}
		top_topics: List[Dict[str, Any]] = []
		for rank, topic in enumerate(topics_sorted, start=1):
			qrow = quality_map.get(topic['name'], {})
			percentage = (100.0 * topic.get('document_count', 0) / max(1, document_count)) if document_count else 0.0
			top_topics.append({
				'rank': rank,
				'name': topic['name'],
				'document_count': topic.get('document_count', 0),
				'document_percentage': round(percentage, 1),
				'quality_score': float(qrow.get('quality_score', 0.0)),
				'top_keywords': dict(list(topic.get('keywords', {}).items())[:5]),
			})

		return {
			'summary': summary,
			'extraction_statistics': stats,
			'top_topics': top_topics,
		}

	def save_results_to_excel(self, doc_topic_dist: pd.DataFrame, topic_word_dist: pd.DataFrame, quality_df: pd.DataFrame, final_topics: List[Dict[str, Any]], report: Dict[str, Any]) -> None:
		try:
			excel_path = os.path.join(self.output_folder, 'topic_analysis_results.xlsx')
			with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
				if not doc_topic_dist.empty:
					doc_topic_dist.to_excel(writer, sheet_name='Document-Topic Distribution')
				if not topic_word_dist.empty:
					topic_word_dist.to_excel(writer, sheet_name='Topic-Word Distribution')
				if not quality_df.empty:
					quality_df.to_excel(writer, sheet_name='Topic Quality Analysis', index=False)
				if final_topics:
					try:
						df_final = pd.DataFrame(final_topics)
						df_final.to_excel(writer, sheet_name='Final Topics', index=False)
					except Exception as exc:
						logger.warning(f"Could not write final_topics to Excel: {exc}")
				if report and 'summary' in report:
					try:
						df_summary = pd.json_normalize(report['summary'])
						df_summary.to_excel(writer, sheet_name='Summary', index=False)
					except Exception as exc:
						logger.warning(f"Could not write report summary to Excel: {exc}")
			logger.info(f"✅ Excel report saved to {excel_path}")
		except Exception as exc:
			logger.error(f"Failed to save Excel report: {exc}")

	def print_summary(self, report: Dict[str, Any]) -> None:
		mode_text = "TEST MODE" if self.test_mode else "FULL"
		print("\n" + "=" * 80)
		print(f"INTELLIGENT TOPIC ANALYSIS COMPLETE! ({mode_text})")
		print("=" * 80)
		print(f"\n📊 Summary Statistics:")
		print(f"   - Total Documents: {report['summary']['total_documents']}")
		print(f"   - Total Extracted Topics: {report['summary']['total_extracted_topics']}")
		print(f"   - Final Merged Topics: {report['summary']['final_merged_topics']}")
		print(f"   - Vocabulary Size: {report['summary']['vocabulary_size']}")
		print(f"\n📈 Extraction Statistics:")
		stats = report['extraction_statistics']
		print(f"   - Avg Topics per Document: {stats['avg_topics_per_doc']:.2f} ± {stats['std_topics_per_doc']:.2f}")
		print(f"   - Topic Count Range: {stats['min_topics_per_doc']} - {stats['max_topics_per_doc']}")
		print(f"\n🏆 Top 10 Topics:")
		for topic in report['top_topics'][:10]:
			print(f"\n   {topic['rank']}. {topic['name']}")
			print(f"      - Documents: {topic['document_count']} ({topic['document_percentage']:.1f}%)")
			print(f"      - Quality Score: {topic['quality_score']:.3f}")
			print(f"      - Top Keywords: {', '.join(list(topic['top_keywords'].keys())[:5])}")
		print(f"\n💾 Output Files:")
		output_dir = f"{self.output_folder}/"
		print(f"   Location: {output_dir}")
		print(f"   - Document-Topic Distribution: doc_topic_distribution.csv")
		print(f"   - Topic-Word Distribution: topic_word_distribution.csv")
		print(f"   - Topic Quality Analysis: topic_quality_analysis.csv")
		print(f"   - Final Topics: final_topics.json")
		print(f"   - Alignment: merge_alignment.csv")
		print(f"   - Analysis Report: analysis_report.json")
		print(f"   - Excel Report: topic_analysis_results.xlsx")
		print("\n" + "=" * 80)


# ---------- CLI ----------

def _read_api_key_from_file(path: str) -> Optional[str]:
	try:
		with open(path, 'r', encoding='utf-8') as fh:
			return fh.read().strip()
	except Exception:
		return None


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description="Topic modeling via OpenAI with lexical merging (no embeddings)")
	parser.add_argument('--api-key', type=str, default=os.getenv('OPENAI_API_KEY'), help='OpenAI API key or path to a file containing it')
	parser.add_argument('--api-key-file', type=str, default=None, help='Path to a file containing OpenAI API key')
	parser.add_argument('--input', type=str, required=True, help='Folder with transcript .txt files')
	parser.add_argument('--output', type=str, required=True, help='Output folder')
	parser.add_argument('--test-limit', type=int, default=None, help='Process only first N files')
	parser.add_argument('--test-mode', action='store_true', help='Write to test_output subfolder')
	parser.add_argument('--similarity-threshold', type=float, default=None, help='Override lexical merge threshold (default 0.6)')
	args = parser.parse_args()

	api_key = args.api_key
	if args.api_key_file:
		api_key_from_file = _read_api_key_from_file(args.api_key_file)
		if api_key_from_file:
			api_key = api_key_from_file

	config: Dict[str, Any] = {}
	if args.similarity_threshold is not None:
		config['similarity_threshold'] = args.similarity_threshold

	pipeline = ComprehensiveAnalysisPipeline(api_key, args.input, args.output, test_mode=args.test_mode, config=config)
	pipeline.run_analysis(test_limit=args.test_limit)


if __name__ == '__main__':
	main()