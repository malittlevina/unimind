"""
hierarchical_memory.py – Hierarchical memory system for Unimind.
Provides short-term, working, long-term, and episodic memory with automatic consolidation.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
import threading

class MemoryLevel(Enum):
    """Hierarchical memory levels."""
    SENSORY = "sensory"      # Immediate sensory input (milliseconds)
    SHORT_TERM = "short_term"  # Working memory (seconds to minutes)
    WORKING = "working"      # Active processing (minutes to hours)
    LONG_TERM = "long_term"  # Persistent knowledge (hours to years)
    EPISODIC = "episodic"    # Personal experiences (lifetime)

class MemoryType(Enum):
    """Types of memory content."""
    FACT = "fact"           # Declarative facts
    PROCEDURE = "procedure" # How-to knowledge
    CONCEPT = "concept"     # Abstract concepts
    EXPERIENCE = "experience" # Personal experiences
    EMOTION = "emotion"     # Emotional associations
    RELATIONSHIP = "relationship" # Connections between items

@dataclass
class MemoryItem:
    """Represents a single memory item."""
    id: str
    content: Any
    memory_type: MemoryType
    memory_level: MemoryLevel
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)  # IDs of related memories
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "memory_level": self.memory_level.value,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
            "confidence": self.confidence,
            "tags": self.tags,
            "associations": self.associations,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            memory_level=MemoryLevel(data["memory_level"]),
            timestamp=data["timestamp"],
            access_count=data["access_count"],
            last_accessed=data["last_accessed"],
            importance=data["importance"],
            confidence=data["confidence"],
            tags=data["tags"],
            associations=data["associations"],
            metadata=data["metadata"]
        )

@dataclass
class MemoryQuery:
    """Represents a memory query."""
    query: str
    memory_types: Optional[List[MemoryType]] = None
    memory_levels: Optional[List[MemoryLevel]] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    min_confidence: float = 0.0
    limit: Optional[int] = None
    time_range: Optional[Tuple[float, float]] = None

@dataclass
class MemoryConsolidation:
    """Represents memory consolidation results."""
    source_memories: List[str]
    consolidated_memory: MemoryItem
    consolidation_type: str
    confidence: float
    reasoning: str

class HierarchicalMemory:
    """
    Hierarchical memory system with automatic consolidation and retrieval optimization.
    """
    
    def __init__(self, db_path: str = "unimind_memory.db"):
        """Initialize the hierarchical memory system."""
        self.logger = logging.getLogger('HierarchicalMemory')
        self.db_path = db_path
        
        # Memory storage by level
        self.memories: Dict[MemoryLevel, Dict[str, MemoryItem]] = {
            level: {} for level in MemoryLevel
        }
        
        # Index for fast retrieval
        self.content_index: Dict[str, List[str]] = {}  # content_hash -> memory_ids
        self.tag_index: Dict[str, List[str]] = {}      # tag -> memory_ids
        self.association_index: Dict[str, List[str]] = {}  # memory_id -> associated_ids
        
        # Configuration
        self.max_sensory_memories = 1000
        self.max_short_term_memories = 500
        self.max_working_memories = 200
        self.max_long_term_memories = 10000
        self.max_episodic_memories = 5000
        
        # Consolidation settings
        self.consolidation_threshold = 0.7
        self.importance_decay_rate = 0.1
        self.access_boost_factor = 0.05
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("Hierarchical memory system initialized")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        memory_type TEXT,
                        memory_level TEXT,
                        timestamp REAL,
                        access_count INTEGER,
                        last_accessed REAL,
                        importance REAL,
                        confidence REAL,
                        tags TEXT,
                        associations TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_level ON memories(memory_level)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
                
                # Load existing memories
                self._load_memories_from_db()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_memories_from_db(self):
        """Load memories from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories")
                for row in cursor.fetchall():
                    memory_data = {
                        "id": row[0],
                        "content": json.loads(row[1]),
                        "memory_type": row[2],
                        "memory_level": row[3],
                        "timestamp": row[4],
                        "access_count": row[5],
                        "last_accessed": row[6],
                        "importance": row[7],
                        "confidence": row[8],
                        "tags": json.loads(row[9]),
                        "associations": json.loads(row[10]),
                        "metadata": json.loads(row[11])
                    }
                    
                    memory_item = MemoryItem.from_dict(memory_data)
                    self._add_memory_to_indexes(memory_item)
                    
        except Exception as e:
            self.logger.error(f"Failed to load memories from database: {e}")
    
    def _add_memory_to_indexes(self, memory: MemoryItem):
        """Add memory to all indexes."""
        # Add to level storage
        self.memories[memory.memory_level][memory.id] = memory
        
        # Add to content index
        content_hash = self._hash_content(memory.content)
        if content_hash not in self.content_index:
            self.content_index[content_hash] = []
        self.content_index[content_hash].append(memory.id)
        
        # Add to tag index
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(memory.id)
        
        # Add to association index
        self.association_index[memory.id] = memory.associations
    
    def store(self, content: Any, memory_type: MemoryType, 
              memory_level: MemoryLevel = MemoryLevel.SHORT_TERM,
              importance: float = 0.5, confidence: float = 0.5,
              tags: List[str] = None, associations: List[str] = None,
              metadata: Dict[str, Any] = None) -> str:
        """
        Store a new memory item.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            memory_level: Memory level
            importance: Importance score (0.0 to 1.0)
            confidence: Confidence score (0.0 to 1.0)
            tags: Optional tags
            associations: Optional associated memory IDs
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        with self.lock:
            # Generate unique ID
            memory_id = self._generate_memory_id(content, memory_type, memory_level)
            
            # Create memory item
            memory = MemoryItem(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                memory_level=memory_level,
                timestamp=time.time(),
                importance=importance,
                confidence=confidence,
                tags=tags or [],
                associations=associations or [],
                metadata=metadata or {}
            )
            
            # Check capacity limits
            self._enforce_capacity_limits(memory_level)
            
            # Add to storage and indexes
            self._add_memory_to_indexes(memory)
            
            # Save to database
            self._save_memory_to_db(memory)
            
            self.logger.debug(f"Stored memory: {memory_id} ({memory_type.value} at {memory_level.value})")
            return memory_id
    
    def retrieve(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Retrieve memories based on query.
        
        Args:
            query: Memory query specification
            
        Returns:
            List of matching memory items
        """
        with self.lock:
            candidates = []
            
            # Determine which levels to search
            levels_to_search = query.memory_levels or list(MemoryLevel)
            
            for level in levels_to_search:
                level_memories = self.memories[level].values()
                
                for memory in level_memories:
                    if self._matches_query(memory, query):
                        candidates.append(memory)
            
            # Sort by relevance (importance + recency + access frequency)
            candidates.sort(key=lambda m: self._calculate_relevance(m), reverse=True)
            
            # Apply limit
            if query.limit:
                candidates = candidates[:query.limit]
            
            # Update access statistics
            for memory in candidates:
                memory.access_count += 1
                memory.last_accessed = time.time()
                self._update_memory_importance(memory)
            
            return candidates
    
    def _matches_query(self, memory: MemoryItem, query: MemoryQuery) -> bool:
        """Check if memory matches query criteria."""
        # Check memory types
        if query.memory_types and memory.memory_type not in query.memory_types:
            return False
        
        # Check memory levels
        if query.memory_levels and memory.memory_level not in query.memory_levels:
            return False
        
        # Check importance
        if memory.importance < query.min_importance:
            return False
        
        # Check confidence
        if memory.confidence < query.min_confidence:
            return False
        
        # Check tags
        if query.tags and not any(tag in memory.tags for tag in query.tags):
            return False
        
        # Check time range
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= memory.timestamp <= end_time):
                return False
        
        return True
    
    def _calculate_relevance(self, memory: MemoryItem) -> float:
        """Calculate relevance score for memory."""
        # Base relevance is importance
        relevance = memory.importance
        
        # Boost for recent access
        time_since_access = time.time() - memory.last_accessed
        access_boost = max(0, 1.0 - (time_since_access / 3600)) * self.access_boost_factor
        relevance += access_boost
        
        # Boost for high confidence
        relevance += memory.confidence * 0.1
        
        return relevance
    
    def consolidate_memories(self, memory_ids: List[str]) -> Optional[MemoryConsolidation]:
        """
        Consolidate multiple memories into a single, higher-level memory.
        
        Args:
            memory_ids: IDs of memories to consolidate
            
        Returns:
            Consolidation result or None if consolidation fails
        """
        with self.lock:
            # Retrieve memories
            memories = []
            for memory_id in memory_ids:
                for level_memories in self.memories.values():
                    if memory_id in level_memories:
                        memories.append(level_memories[memory_id])
                        break
            
            if len(memories) < 2:
                return None
            
            # Analyze for consolidation patterns
            consolidation_type = self._analyze_consolidation_pattern(memories)
            
            if not consolidation_type:
                return None
            
            # Create consolidated memory
            consolidated_content = self._create_consolidated_content(memories, consolidation_type)
            consolidated_importance = max(m.importance for m in memories) * 1.1
            consolidated_confidence = sum(m.confidence for m in memories) / len(memories)
            
            # Determine target level (one level higher than source memories)
            source_levels = {m.memory_level for m in memories}
            target_level = self._get_next_level(min(source_levels))
            
            # Create consolidated memory
            consolidated_memory = MemoryItem(
                id=self._generate_memory_id(consolidated_content, MemoryType.CONCEPT, target_level),
                content=consolidated_content,
                memory_type=MemoryType.CONCEPT,
                memory_level=target_level,
                timestamp=time.time(),
                importance=consolidated_importance,
                confidence=consolidated_confidence,
                tags=list(set(tag for m in memories for tag in m.tags)),
                associations=memory_ids,
                metadata={"consolidation_type": consolidation_type, "source_count": len(memories)}
            )
            
            # Store consolidated memory
            self._add_memory_to_indexes(consolidated_memory)
            self._save_memory_to_db(consolidated_memory)
            
            # Create consolidation result
            result = MemoryConsolidation(
                source_memories=memory_ids,
                consolidated_memory=consolidated_memory,
                consolidation_type=consolidation_type,
                confidence=consolidated_confidence,
                reasoning=f"Consolidated {len(memories)} memories into {consolidation_type}"
            )
            
            self.logger.info(f"Consolidated {len(memories)} memories into {consolidated_memory.id}")
            return result
    
    def _analyze_consolidation_pattern(self, memories: List[MemoryItem]) -> Optional[str]:
        """Analyze memories to determine consolidation pattern."""
        # Check for factual consolidation
        if all(m.memory_type == MemoryType.FACT for m in memories):
            return "factual_synthesis"
        
        # Check for procedural consolidation
        if all(m.memory_type == MemoryType.PROCEDURE for m in memories):
            return "procedural_abstraction"
        
        # Check for conceptual consolidation
        if all(m.memory_type == MemoryType.CONCEPT for m in memories):
            return "conceptual_integration"
        
        # Check for experiential consolidation
        if all(m.memory_type == MemoryType.EXPERIENCE for m in memories):
            return "experiential_synthesis"
        
        return None
    
    def _create_consolidated_content(self, memories: List[MemoryItem], consolidation_type: str) -> Any:
        """Create consolidated content based on consolidation type."""
        if consolidation_type == "factual_synthesis":
            # Combine related facts
            facts = [str(m.content) for m in memories]
            return {
                "type": "synthesized_facts",
                "facts": facts,
                "summary": f"Combined {len(facts)} related facts",
                "timestamp": time.time()
            }
        
        elif consolidation_type == "procedural_abstraction":
            # Abstract common procedures
            procedures = [str(m.content) for m in memories]
            return {
                "type": "abstracted_procedure",
                "procedures": procedures,
                "common_pattern": "Extracted common procedural pattern",
                "timestamp": time.time()
            }
        
        elif consolidation_type == "conceptual_integration":
            # Integrate related concepts
            concepts = [str(m.content) for m in memories]
            return {
                "type": "integrated_concept",
                "concepts": concepts,
                "integration": "Integrated related conceptual knowledge",
                "timestamp": time.time()
            }
        
        else:
            # Generic consolidation
            return {
                "type": "consolidated_memory",
                "source_memories": [m.content for m in memories],
                "consolidation_type": consolidation_type,
                "timestamp": time.time()
            }
    
    def _get_next_level(self, current_level: MemoryLevel) -> MemoryLevel:
        """Get the next higher memory level."""
        level_order = [
            MemoryLevel.SENSORY,
            MemoryLevel.SHORT_TERM,
            MemoryLevel.WORKING,
            MemoryLevel.LONG_TERM,
            MemoryLevel.EPISODIC
        ]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return MemoryLevel.LONG_TERM  # Default fallback
    
    def _enforce_capacity_limits(self, memory_level: MemoryLevel):
        """Enforce capacity limits for memory levels."""
        level_limits = {
            MemoryLevel.SENSORY: self.max_sensory_memories,
            MemoryLevel.SHORT_TERM: self.max_short_term_memories,
            MemoryLevel.WORKING: self.max_working_memories,
            MemoryLevel.LONG_TERM: self.max_long_term_memories,
            MemoryLevel.EPISODIC: self.max_episodic_memories
        }
        
        limit = level_limits.get(memory_level)
        if not limit:
            return
        
        current_memories = self.memories[memory_level]
        if len(current_memories) >= limit:
            # Remove least relevant memories
            memories_to_remove = sorted(
                current_memories.values(),
                key=lambda m: self._calculate_relevance(m)
            )[:len(current_memories) - limit + 1]
            
            for memory in memories_to_remove:
                self._remove_memory(memory.id)
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory from all storage and indexes."""
        # Find and remove from level storage
        for level_memories in self.memories.values():
            if memory_id in level_memories:
                del level_memories[memory_id]
                break
        
        # Remove from indexes
        for content_hash, memory_ids in self.content_index.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
                if not memory_ids:
                    del self.content_index[content_hash]
        
        for tag, memory_ids in self.tag_index.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
                if not memory_ids:
                    del self.tag_index[tag]
        
        if memory_id in self.association_index:
            del self.association_index[memory_id]
        
        # Remove from database
        self._remove_memory_from_db(memory_id)
    
    def _update_memory_importance(self, memory: MemoryItem):
        """Update memory importance based on access patterns."""
        # Boost importance for frequently accessed memories
        if memory.access_count > 10:
            memory.importance = min(1.0, memory.importance + 0.1)
        
        # Decay importance over time
        time_since_creation = time.time() - memory.timestamp
        decay_factor = time_since_creation / (24 * 3600)  # Days since creation
        memory.importance = max(0.1, memory.importance - (decay_factor * self.importance_decay_rate))
        
        # Update in database
        self._update_memory_in_db(memory)
    
    def _generate_memory_id(self, content: Any, memory_type: MemoryType, memory_level: MemoryLevel) -> str:
        """Generate unique memory ID."""
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{content_str}:{memory_type.value}:{memory_level.value}:{time.time()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _hash_content(self, content: Any) -> str:
        """Generate hash for content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _save_memory_to_db(self, memory: MemoryItem):
        """Save memory to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_type, memory_level, timestamp, access_count, 
                     last_accessed, importance, confidence, tags, associations, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    json.dumps(memory.content),
                    memory.memory_type.value,
                    memory.memory_level.value,
                    memory.timestamp,
                    memory.access_count,
                    memory.last_accessed,
                    memory.importance,
                    memory.confidence,
                    json.dumps(memory.tags),
                    json.dumps(memory.associations),
                    json.dumps(memory.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Failed to save memory to database: {e}")
    
    def _update_memory_in_db(self, memory: MemoryItem):
        """Update memory in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE memories 
                    SET access_count = ?, last_accessed = ?, importance = ?, confidence = ?
                    WHERE id = ?
                """, (
                    memory.access_count,
                    memory.last_accessed,
                    memory.importance,
                    memory.confidence,
                    memory.id
                ))
        except Exception as e:
            self.logger.error(f"Failed to update memory in database: {e}")
    
    def _remove_memory_from_db(self, memory_id: str):
        """Remove memory from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        except Exception as e:
            self.logger.error(f"Failed to remove memory from database: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for memory management."""
        # Start consolidation thread
        consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        consolidation_thread.start()
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _consolidation_worker(self):
        """Background worker for memory consolidation."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._perform_automatic_consolidation()
            except Exception as e:
                self.logger.error(f"Consolidation worker error: {e}")
    
    def _cleanup_worker(self):
        """Background worker for memory cleanup."""
        while True:
            try:
                time.sleep(600)  # Run every 10 minutes
                self._perform_cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
    
    def _perform_automatic_consolidation(self):
        """Perform automatic memory consolidation."""
        with self.lock:
            # Find memories that might benefit from consolidation
            for level in [MemoryLevel.SHORT_TERM, MemoryLevel.WORKING]:
                memories = list(self.memories[level].values())
                
                # Group similar memories
                groups = self._group_similar_memories(memories)
                
                # Consolidate groups with high similarity
                for group in groups:
                    if len(group) >= 2 and self._calculate_group_similarity(group) > self.consolidation_threshold:
                        memory_ids = [m.id for m in group]
                        self.consolidate_memories(memory_ids)
    
    def _group_similar_memories(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Group memories by similarity."""
        groups = []
        used = set()
        
        for i, memory1 in enumerate(memories):
            if i in used:
                continue
            
            group = [memory1]
            used.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                if self._calculate_similarity(memory1, memory2) > 0.5:
                    group.append(memory2)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories."""
        # Check memory type
        if memory1.memory_type != memory2.memory_type:
            return 0.0
        
        # Check tags overlap
        tag_overlap = len(set(memory1.tags) & set(memory2.tags))
        tag_similarity = tag_overlap / max(len(memory1.tags), len(memory2.tags), 1)
        
        # Check content similarity (simplified)
        content_similarity = 0.5 if str(memory1.content) == str(memory2.content) else 0.0
        
        return (tag_similarity + content_similarity) / 2
    
    def _calculate_group_similarity(self, group: List[MemoryItem]) -> float:
        """Calculate average similarity within a group."""
        if len(group) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total_similarity += self._calculate_similarity(group[i], group[j])
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _perform_cleanup(self):
        """Perform memory cleanup."""
        with self.lock:
            # Remove expired sensory memories
            current_time = time.time()
            sensory_memories = list(self.memories[MemoryLevel.SENSORY].values())
            
            for memory in sensory_memories:
                if current_time - memory.timestamp > 1.0:  # 1 second TTL for sensory
                    self._remove_memory(memory.id)
            
            # Remove low-importance memories from short-term
            short_term_memories = list(self.memories[MemoryLevel.SHORT_TERM].values())
            for memory in short_term_memories:
                if memory.importance < 0.1 and current_time - memory.last_accessed > 3600:  # 1 hour
                    self._remove_memory(memory.id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self.lock:
            stats = {
                "total_memories": sum(len(memories) for memories in self.memories.values()),
                "memories_by_level": {
                    level.value: len(memories) for level, memories in self.memories.items()
                },
                "memories_by_type": {},
                "average_importance": 0.0,
                "average_confidence": 0.0,
                "total_accesses": 0
            }
            
            all_memories = []
            type_counts = {}
            
            for memories in self.memories.values():
                for memory in memories.values():
                    all_memories.append(memory)
                    type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1
            
            if all_memories:
                stats["memories_by_type"] = type_counts
                stats["average_importance"] = sum(m.importance for m in all_memories) / len(all_memories)
                stats["average_confidence"] = sum(m.confidence for m in all_memories) / len(all_memories)
                stats["total_accesses"] = sum(m.access_count for m in all_memories)
            
            return stats

# Global hierarchical memory instance
hierarchical_memory = HierarchicalMemory()

def store_memory(content: Any, memory_type: MemoryType, 
                memory_level: MemoryLevel = MemoryLevel.SHORT_TERM,
                importance: float = 0.5, confidence: float = 0.5,
                tags: List[str] = None, associations: List[str] = None,
                metadata: Dict[str, Any] = None) -> str:
    """Store a memory using the global hierarchical memory instance."""
    return hierarchical_memory.store(content, memory_type, memory_level, importance, 
                                   confidence, tags, associations, metadata)

def retrieve_memories(query: MemoryQuery) -> List[MemoryItem]:
    """Retrieve memories using the global hierarchical memory instance."""
    return hierarchical_memory.retrieve(query)

def consolidate_memories(memory_ids: List[str]) -> Optional[MemoryConsolidation]:
    """Consolidate memories using the global hierarchical memory instance."""
    return hierarchical_memory.consolidate_memories(memory_ids)

def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics using the global hierarchical memory instance."""
    return hierarchical_memory.get_memory_stats() 