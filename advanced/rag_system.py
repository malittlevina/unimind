"""
RAG (Retrieval-Augmented Generation) System for UniMind

Advanced retrieval-augmented generation system that enhances responses with
verified information from multiple knowledge sources and memory systems.
"""

import asyncio
import logging
import time
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading
import numpy as np
from pathlib import Path

# External dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    import wikipedia
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class KnowledgeSourceType(Enum):
    """Types of knowledge sources."""
    MEMORY = "memory"
    DOCUMENT = "document"
    WEB = "web"
    WIKIPEDIA = "wikipedia"
    DATABASE = "database"
    API = "api"
    USER_INPUT = "user_input"
    VERIFIED_FACT = "verified_fact"


class RetrievalMethod(Enum):
    """Methods for retrieving relevant information."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCHING = "keyword_matching"
    VECTOR_SIMILARITY = "vector_similarity"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"


class VerificationLevel(Enum):
    """Levels of information verification."""
    UNVERIFIED = "unverified"
    PARTIALLY_VERIFIED = "partially_verified"
    VERIFIED = "verified"
    HIGHLY_VERIFIED = "highly_verified"
    FACT_CHECKED = "fact_checked"


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge with metadata."""
    chunk_id: str
    content: str
    source_type: KnowledgeSourceType
    source_id: str
    confidence: float
    verification_level: VerificationLevel
    timestamp: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result of a knowledge retrieval operation."""
    query: str
    retrieved_chunks: List[KnowledgeChunk]
    relevance_scores: List[float]
    retrieval_method: RetrievalMethod
    total_results: int
    retrieval_time: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Enhanced response with retrieved information."""
    original_query: str
    enhanced_response: str
    retrieved_context: List[KnowledgeChunk]
    verification_summary: Dict[str, Any]
    confidence: float
    sources_used: List[str]
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGSystem:
    """
    Advanced RAG system for UniMind that enhances responses with verified information.
    
    Features:
    - Multi-source knowledge retrieval
    - Information verification and fact-checking
    - Semantic similarity search
    - Context-aware response generation
    - Confidence scoring
    - Source attribution
    """
    
    def __init__(self, 
                 memory_system=None,
                 llm_engine=None,
                 knowledge_base_path: str = "unimind_knowledge_base"):
        """Initialize the RAG system."""
        self.logger = logging.getLogger('RAGSystem')
        
        # Core systems
        self.memory_system = memory_system
        self.llm_engine = llm_engine
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Knowledge storage
        self.knowledge_chunks: Dict[str, KnowledgeChunk] = {}
        self.chunk_index: Dict[str, List[str]] = defaultdict(list)  # tag -> chunk_ids
        self.source_index: Dict[str, List[str]] = defaultdict(list)  # source -> chunk_ids
        
        # Embedding system
        self.embedding_model = None
        self.chunk_embeddings: Dict[str, List[float]] = {}
        self.embedding_dimension = 768
        
        # Verification system
        self.verification_sources = self._load_verification_sources()
        self.fact_checking_rules = self._load_fact_checking_rules()
        
        # Configuration
        self.max_retrieval_results = 10
        self.min_relevance_threshold = 0.3
        self.max_context_length = 4000
        self.enable_web_search = REQUESTS_AVAILABLE
        self.enable_wikipedia = REQUESTS_AVAILABLE
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize_embedding_model()
        self._load_existing_knowledge()
        
        self.logger.info("RAG system initialized")
    
    def _load_verification_sources(self) -> Dict[str, str]:
        """Load verification sources for fact-checking."""
        return {
            'wikipedia': 'https://en.wikipedia.org/wiki/',
            'fact_check': 'https://www.factcheck.org/',
            'snopes': 'https://www.snopes.com/',
            'reuters_fact_check': 'https://www.reuters.com/fact-check/',
            'ap_fact_check': 'https://apnews.com/hub/fact-checking'
        }
    
    def _load_fact_checking_rules(self) -> Dict[str, Any]:
        """Load fact-checking rules and patterns."""
        return {
            'claim_patterns': [
                r'(\w+)\s+(?:claims?|says?|stated?|reported?)\s+(?:that\s+)?(.+)',
                r'(?:According\s+to|Based\s+on)\s+(\w+)[,\s]+(.+)',
                r'(\w+)\s+(?:announced?|revealed?|discovered?)\s+(?:that\s+)?(.+)'
            ],
            'verification_keywords': [
                'verified', 'confirmed', 'proven', 'established', 'documented',
                'factual', 'accurate', 'reliable', 'trustworthy', 'authentic'
            ],
            'uncertainty_keywords': [
                'allegedly', 'reportedly', 'supposedly', 'rumored', 'claimed',
                'unverified', 'unconfirmed', 'uncertain', 'doubtful', 'questionable'
            ]
        }
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic search."""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
                self.logger.info(f"Embedding model initialized with dimension {self.embedding_dimension}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        else:
            self.logger.warning("Transformers not available - using keyword matching only")
    
    def _load_existing_knowledge(self):
        """Load existing knowledge from storage."""
        knowledge_file = self.knowledge_base_path / "knowledge_chunks.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    data = json.load(f)
                    for chunk_data in data:
                        chunk = KnowledgeChunk(**chunk_data)
                        self.knowledge_chunks[chunk.chunk_id] = chunk
                        self._index_chunk(chunk)
                self.logger.info(f"Loaded {len(self.knowledge_chunks)} knowledge chunks")
            except Exception as e:
                self.logger.error(f"Failed to load existing knowledge: {e}")
    
    def _save_knowledge(self):
        """Save knowledge chunks to storage."""
        knowledge_file = self.knowledge_base_path / "knowledge_chunks.json"
        try:
            # Convert chunks to serializable format
            serializable_chunks = []
            for chunk in self.knowledge_chunks.values():
                chunk_dict = asdict(chunk)
                # Convert Enum values to strings
                chunk_dict['source_type'] = chunk.source_type.value
                chunk_dict['verification_level'] = chunk.verification_level.value
                serializable_chunks.append(chunk_dict)
            
            with open(knowledge_file, 'w') as f:
                json.dump(serializable_chunks, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
    
    def _index_chunk(self, chunk: KnowledgeChunk):
        """Index a knowledge chunk for retrieval."""
        # Index by tags
        for tag in chunk.tags:
            self.chunk_index[tag].append(chunk.chunk_id)
        
        # Index by source
        self.source_index[chunk.source_id].append(chunk.chunk_id)
        
        # Generate embedding if model is available
        if self.embedding_model and chunk.embedding is None:
            try:
                chunk.embedding = self.embedding_model.encode(chunk.content).tolist()
                self.chunk_embeddings[chunk.chunk_id] = chunk.embedding
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
    
    async def add_knowledge(self, 
                          content: str,
                          source_type: KnowledgeSourceType,
                          source_id: str,
                          confidence: float = 0.8,
                          tags: List[str] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """Add knowledge to the RAG system."""
        chunk_id = self._generate_chunk_id(content, source_id)
        
        # Determine verification level
        verification_level = self._determine_verification_level(source_type, confidence)
        
        # Create knowledge chunk
        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            content=content,
            source_type=source_type,
            source_id=source_id,
            confidence=confidence,
            verification_level=verification_level,
            timestamp=time.time(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store and index
        with self.lock:
            self.knowledge_chunks[chunk_id] = chunk
            self._index_chunk(chunk)
            self._save_knowledge()
        
        self.logger.info(f"Added knowledge chunk: {chunk_id}")
        return chunk_id
    
    def _generate_chunk_id(self, content: str, source_id: str) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"{source_id}_{content_hash}_{timestamp}"
    
    def _determine_verification_level(self, source_type: KnowledgeSourceType, confidence: float) -> VerificationLevel:
        """Determine verification level based on source type and confidence."""
        if source_type == KnowledgeSourceType.VERIFIED_FACT:
            return VerificationLevel.FACT_CHECKED
        elif source_type == KnowledgeSourceType.WIKIPEDIA:
            return VerificationLevel.VERIFIED if confidence > 0.8 else VerificationLevel.PARTIALLY_VERIFIED
        elif source_type == KnowledgeSourceType.WEB:
            return VerificationLevel.PARTIALLY_VERIFIED if confidence > 0.7 else VerificationLevel.UNVERIFIED
        elif source_type == KnowledgeSourceType.MEMORY:
            return VerificationLevel.PARTIALLY_VERIFIED if confidence > 0.6 else VerificationLevel.UNVERIFIED
        else:
            return VerificationLevel.UNVERIFIED
    
    async def retrieve_knowledge(self, 
                               query: str,
                               method: RetrievalMethod = RetrievalMethod.HYBRID,
                               max_results: int = None,
                               min_confidence: float = None) -> RetrievalResult:
        """Retrieve relevant knowledge for a query."""
        start_time = time.time()
        max_results = max_results or self.max_retrieval_results
        min_confidence = min_confidence or self.min_relevance_threshold
        
        self.logger.info(f"Retrieving knowledge for query: {query}")
        
        # Perform retrieval based on method
        if method == RetrievalMethod.SEMANTIC_SIMILARITY:
            chunks, scores = await self._semantic_search(query, max_results)
        elif method == RetrievalMethod.KEYWORD_MATCHING:
            chunks, scores = await self._keyword_search(query, max_results)
        elif method == RetrievalMethod.VECTOR_SIMILARITY:
            chunks, scores = await self._vector_search(query, max_results)
        elif method == RetrievalMethod.CONTEXTUAL:
            chunks, scores = await self._contextual_search(query, max_results)
        else:  # HYBRID
            chunks, scores = await self._hybrid_search(query, max_results)
        
        # Filter by confidence and relevance
        filtered_results = []
        filtered_scores = []
        
        for chunk, score in zip(chunks, scores):
            if score >= min_confidence and chunk.confidence >= min_confidence:
                filtered_results.append(chunk)
                filtered_scores.append(score)
        
        # Sort by relevance score
        sorted_pairs = sorted(zip(filtered_results, filtered_scores), 
                            key=lambda x: x[1], reverse=True)
        
        if sorted_pairs:
            filtered_results, filtered_scores = zip(*sorted_pairs)
        else:
            filtered_results, filtered_scores = [], []
        
        # Limit results
        filtered_results = filtered_results[:max_results]
        filtered_scores = filtered_scores[:max_results]
        
        # Calculate overall confidence
        overall_confidence = np.mean(filtered_scores) if filtered_scores else 0.0
        
        result = RetrievalResult(
            query=query,
            retrieved_chunks=list(filtered_results),
            relevance_scores=list(filtered_scores),
            retrieval_method=method,
            total_results=len(filtered_results),
            retrieval_time=time.time() - start_time,
            confidence=overall_confidence,
            metadata={
                'method': method.value,
                'min_confidence': min_confidence,
                'max_results': max_results
            }
        )
        
        self.logger.info(f"Retrieved {len(filtered_results)} chunks in {result.retrieval_time:.3f}s")
        return result
    
    async def _semantic_search(self, query: str, max_results: int) -> Tuple[List[KnowledgeChunk], List[float]]:
        """Perform semantic similarity search."""
        if not self.embedding_model:
            return [], []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            similarities = []
            chunks = []
            
            for chunk_id, chunk in self.knowledge_chunks.items():
                if chunk.embedding:
                    chunk_embedding = np.array(chunk.embedding)
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    similarities.append(similarity)
                    chunks.append(chunk)
            
            # Sort by similarity
            sorted_pairs = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
            
            if sorted_pairs:
                chunks, similarities = zip(*sorted_pairs)
                return list(chunks[:max_results]), list(similarities[:max_results])
            else:
                return [], []
                
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return [], []
    
    async def _keyword_search(self, query: str, max_results: int) -> Tuple[List[KnowledgeChunk], List[float]]:
        """Perform keyword-based search."""
        query_words = set(query.lower().split())
        results = []
        scores = []
        
        for chunk in self.knowledge_chunks.values():
            chunk_words = set(chunk.content.lower().split())
            chunk_tags = set(tag.lower() for tag in chunk.tags)
            
            # Calculate keyword overlap
            content_overlap = len(query_words & chunk_words) / len(query_words) if query_words else 0
            tag_overlap = len(query_words & chunk_tags) / len(query_words) if query_words else 0
            
            # Combined score
            score = content_overlap * 0.7 + tag_overlap * 0.3
            
            if score > 0:
                results.append(chunk)
                scores.append(score)
        
        # Sort by score
        sorted_pairs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        
        if sorted_pairs:
            results, scores = zip(*sorted_pairs)
            return list(results[:max_results]), list(scores[:max_results])
        else:
            return [], []
    
    async def _vector_search(self, query: str, max_results: int) -> Tuple[List[KnowledgeChunk], List[float]]:
        """Perform vector similarity search."""
        # Similar to semantic search but with more sophisticated vector operations
        return await self._semantic_search(query, max_results)
    
    async def _contextual_search(self, query: str, max_results: int) -> Tuple[List[KnowledgeChunk], List[float]]:
        """Perform context-aware search."""
        # Combine multiple search methods with context consideration
        semantic_chunks, semantic_scores = await self._semantic_search(query, max_results)
        keyword_chunks, keyword_scores = await self._keyword_search(query, max_results)
        
        # Merge and deduplicate results
        all_chunks = {}
        all_scores = {}
        
        for chunk, score in zip(semantic_chunks, semantic_scores):
            all_chunks[chunk.chunk_id] = chunk
            all_scores[chunk.chunk_id] = score * 0.6  # Weight semantic search
        
        for chunk, score in zip(keyword_chunks, keyword_scores):
            if chunk.chunk_id in all_chunks:
                all_scores[chunk.chunk_id] += score * 0.4  # Add keyword weight
            else:
                all_chunks[chunk.chunk_id] = chunk
                all_scores[chunk.chunk_id] = score * 0.4
        
        # Sort by combined scores
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        chunks = [all_chunks[chunk_id] for chunk_id, _ in sorted_items[:max_results]]
        scores = [score for _, score in sorted_items[:max_results]]
        
        return chunks, scores
    
    async def _hybrid_search(self, query: str, max_results: int) -> Tuple[List[KnowledgeChunk], List[float]]:
        """Perform hybrid search combining multiple methods."""
        return await self._contextual_search(query, max_results)
    
    async def generate_rag_response(self, 
                                  query: str,
                                  base_response: str = None,
                                  retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
                                  enhance_response: bool = True) -> RAGResponse:
        """Generate an enhanced response using RAG."""
        start_time = time.time()
        
        self.logger.info(f"Generating RAG response for: {query}")
        
        # Retrieve relevant knowledge
        retrieval_result = await self.retrieve_knowledge(query, retrieval_method)
        
        # Generate enhanced response
        if enhance_response and retrieval_result.retrieved_chunks:
            enhanced_response = await self._enhance_response_with_context(
                query, base_response, retrieval_result.retrieved_chunks
            )
        else:
            enhanced_response = base_response or "I don't have enough verified information to answer that question."
        
        # Create verification summary
        verification_summary = self._create_verification_summary(retrieval_result.retrieved_chunks)
        
        # Extract sources
        sources_used = list(set(chunk.source_id for chunk in retrieval_result.retrieved_chunks))
        
        response = RAGResponse(
            original_query=query,
            enhanced_response=enhanced_response,
            retrieved_context=retrieval_result.retrieved_chunks,
            verification_summary=verification_summary,
            confidence=retrieval_result.confidence,
            sources_used=sources_used,
            generation_time=time.time() - start_time,
            metadata={
                'retrieval_method': retrieval_method.value,
                'chunks_retrieved': len(retrieval_result.retrieved_chunks),
                'enhancement_applied': enhance_response
            }
        )
        
        self.logger.info(f"Generated RAG response in {response.generation_time:.3f}s")
        return response
    
    async def _enhance_response_with_context(self, 
                                           query: str,
                                           base_response: str,
                                           context_chunks: List[KnowledgeChunk]) -> str:
        """Enhance response with retrieved context."""
        if not context_chunks:
            return base_response or "I don't have enough verified information to answer that question."
        
        # Prepare context
        context_text = "\n\n".join([
            f"Source: {chunk.source_id} (Confidence: {chunk.confidence:.2f})\n{chunk.content}"
            for chunk in context_chunks[:5]  # Limit context length
        ])
        
        # Create enhanced prompt
        enhanced_prompt = f"""
Based on the following verified information, please provide a comprehensive answer to the user's question.

User Question: {query}

Verified Information:
{context_text}

Original Response: {base_response or "No base response provided."}

Please enhance the response with the verified information above, ensuring accuracy and providing source attribution where appropriate.
"""
        
        # Generate enhanced response using LLM if available
        if self.llm_engine:
            try:
                enhanced_response = await self.llm_engine.run(enhanced_prompt)
                return enhanced_response
            except Exception as e:
                self.logger.warning(f"Failed to enhance response with LLM: {e}")
        
        # Fallback: simple enhancement
        return self._simple_enhancement(query, base_response, context_chunks)
    
    def _simple_enhancement(self, 
                          query: str,
                          base_response: str,
                          context_chunks: List[KnowledgeChunk]) -> str:
        """Simple enhancement without LLM."""
        if not context_chunks:
            return base_response or "I don't have enough verified information to answer that question."
        
        # Create enhanced response
        enhanced_parts = []
        
        if base_response:
            enhanced_parts.append(f"Based on my knowledge: {base_response}")
        
        enhanced_parts.append("\nVerified Information:")
        for chunk in context_chunks[:3]:  # Limit to top 3 sources
            enhanced_parts.append(f"• {chunk.content} (Source: {chunk.source_id})")
        
        enhanced_parts.append(f"\nConfidence Level: {np.mean([c.confidence for c in context_chunks]):.2f}")
        
        return "\n".join(enhanced_parts)
    
    def _create_verification_summary(self, chunks: List[KnowledgeChunk]) -> Dict[str, Any]:
        """Create a summary of verification information."""
        if not chunks:
            return {"verified": False, "confidence": 0.0, "sources": []}
        
        verification_levels = [chunk.verification_level for chunk in chunks]
        confidences = [chunk.confidence for chunk in chunks]
        sources = list(set(chunk.source_id for chunk in chunks))
        
        # Count verification levels
        level_counts = defaultdict(int)
        for level in verification_levels:
            level_counts[level.value] += 1
        
        return {
            "verified": any(level in [VerificationLevel.VERIFIED, VerificationLevel.HIGHLY_VERIFIED, VerificationLevel.FACT_CHECKED] 
                          for level in verification_levels),
            "confidence": np.mean(confidences),
            "verification_levels": dict(level_counts),
            "sources": sources,
            "total_chunks": len(chunks),
            "highest_verification": max(verification_levels, key=lambda x: x.value).value
        }
    
    async def verify_information(self, claim: str) -> Dict[str, Any]:
        """Verify a specific claim against available knowledge."""
        self.logger.info(f"Verifying claim: {claim}")
        
        # Search for relevant information
        retrieval_result = await self.retrieve_knowledge(claim, RetrievalMethod.HYBRID)
        
        # Analyze claim against retrieved information
        verification_result = {
            "claim": claim,
            "verified": False,
            "confidence": 0.0,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "verification_level": VerificationLevel.UNVERIFIED.value,
            "sources": [],
            "analysis": ""
        }
        
        if retrieval_result.retrieved_chunks:
            # Analyze each chunk for support or contradiction
            for chunk in retrieval_result.retrieved_chunks:
                relevance = self._analyze_claim_relevance(claim, chunk.content)
                
                if relevance > 0.5:  # Relevant chunk
                    if self._supports_claim(claim, chunk.content):
                        verification_result["supporting_evidence"].append({
                            "content": chunk.content,
                            "source": chunk.source_id,
                            "confidence": chunk.confidence,
                            "relevance": relevance
                        })
                    elif self._contradicts_claim(claim, chunk.content):
                        verification_result["contradicting_evidence"].append({
                            "content": chunk.content,
                            "source": chunk.source_id,
                            "confidence": chunk.confidence,
                            "relevance": relevance
                        })
            
            # Determine verification status
            supporting_count = len(verification_result["supporting_evidence"])
            contradicting_count = len(verification_result["contradicting_evidence"])
            
            if supporting_count > contradicting_count:
                verification_result["verified"] = True
                verification_result["confidence"] = min(1.0, supporting_count / (supporting_count + contradicting_count + 1))
            elif contradicting_count > supporting_count:
                verification_result["verified"] = False
                verification_result["confidence"] = min(1.0, contradicting_count / (supporting_count + contradicting_count + 1))
            else:
                verification_result["verified"] = False
                verification_result["confidence"] = 0.5
            
            # Set verification level
            if verification_result["confidence"] > 0.8:
                verification_result["verification_level"] = VerificationLevel.HIGHLY_VERIFIED.value
            elif verification_result["confidence"] > 0.6:
                verification_result["verification_level"] = VerificationLevel.VERIFIED.value
            elif verification_result["confidence"] > 0.4:
                verification_result["verification_level"] = VerificationLevel.PARTIALLY_VERIFIED.value
            
            # Extract sources
            verification_result["sources"] = list(set(
                evidence["source"] for evidence in verification_result["supporting_evidence"] + verification_result["contradicting_evidence"]
            ))
            
            # Generate analysis
            verification_result["analysis"] = self._generate_verification_analysis(verification_result)
        
        return verification_result
    
    def _analyze_claim_relevance(self, claim: str, content: str) -> float:
        """Analyze how relevant content is to a claim."""
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words & content_words)
        return overlap / len(claim_words)
    
    def _supports_claim(self, claim: str, content: str) -> bool:
        """Determine if content supports the claim."""
        # Simple keyword-based support detection
        claim_lower = claim.lower()
        content_lower = content.lower()
        
        # Look for supporting keywords
        supporting_keywords = ['true', 'correct', 'accurate', 'verified', 'confirmed', 'proven', 'supports', 'agrees']
        
        # Check for negation
        negation_keywords = ['not', 'false', 'incorrect', 'wrong', 'disproven', 'contradicts', 'disagrees']
        
        has_supporting = any(keyword in content_lower for keyword in supporting_keywords)
        has_negation = any(keyword in content_lower for keyword in negation_keywords)
        
        return has_supporting and not has_negation
    
    def _contradicts_claim(self, claim: str, content: str) -> bool:
        """Determine if content contradicts the claim."""
        # Simple keyword-based contradiction detection
        content_lower = content.lower()
        
        # Look for contradicting keywords
        contradicting_keywords = ['false', 'incorrect', 'wrong', 'disproven', 'contradicts', 'disagrees', 'not true']
        
        return any(keyword in content_lower for keyword in contradicting_keywords)
    
    def _generate_verification_analysis(self, verification_result: Dict[str, Any]) -> str:
        """Generate a human-readable analysis of verification results."""
        supporting_count = len(verification_result["supporting_evidence"])
        contradicting_count = len(verification_result["contradicting_evidence"])
        confidence = verification_result["confidence"]
        
        if supporting_count > contradicting_count:
            return f"This claim appears to be verified with {confidence:.1%} confidence. Found {supporting_count} supporting pieces of evidence vs {contradicting_count} contradicting pieces."
        elif contradicting_count > supporting_count:
            return f"This claim appears to be false with {confidence:.1%} confidence. Found {contradicting_count} contradicting pieces of evidence vs {supporting_count} supporting pieces."
        else:
            return f"This claim cannot be definitively verified. Found {supporting_count} supporting and {contradicting_count} contradicting pieces of evidence."
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status."""
        return {
            'total_chunks': len(self.knowledge_chunks),
            'embedding_model_available': self.embedding_model is not None,
            'web_search_enabled': self.enable_web_search,
            'wikipedia_enabled': self.enable_wikipedia,
            'verification_sources': len(self.verification_sources),
            'knowledge_base_path': str(self.knowledge_base_path),
            'retrieval_methods': [method.value for method in RetrievalMethod],
            'verification_levels': [level.value for level in VerificationLevel]
        }
    
    async def cleanup_old_knowledge(self, max_age_days: int = 30):
        """Clean up old knowledge chunks."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        with self.lock:
            old_chunks = [
                chunk_id for chunk_id, chunk in self.knowledge_chunks.items()
                if chunk.timestamp < cutoff_time and chunk.verification_level == VerificationLevel.UNVERIFIED
            ]
            
            for chunk_id in old_chunks:
                del self.knowledge_chunks[chunk_id]
                if chunk_id in self.chunk_embeddings:
                    del self.chunk_embeddings[chunk_id]
            
            # Rebuild indexes
            self.chunk_index.clear()
            self.source_index.clear()
            for chunk in self.knowledge_chunks.values():
                self._index_chunk(chunk)
            
            self._save_knowledge()
        
        self.logger.info(f"Cleaned up {len(old_chunks)} old knowledge chunks")


# Global RAG system instance
rag_system = RAGSystem() 