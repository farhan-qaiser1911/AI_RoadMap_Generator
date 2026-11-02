"""
PATHFINDER // Technical Curriculum & Learning Roadmap Architect
Powered by Google Gemini 3.8 Flash (`gemini-3.8-flash`) and Streamlit.

Architecture:
1. Domain Layer: Strongly-typed dataclasses & Pydantic schemas with deterministic JSON validation.
2. Service Layer: Resilient Gemini 3.8 Flash client wrapper (`google-genai` SDK) with structured output.
3. Export Layer: Markdown study pack, Graphviz DOT dependency graph, and raw JSON exporters.
4. Presentation Layer: Custom dark brutalist/editorial UI ("Obsidian & Amber Blueprint") with
   interactive SVG topology graphs, phase cards, resource matrices, and persistent progress tracking.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from pydantic import BaseModel, Field

# Configure structured module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("pathfinder")

# ============================================================================
# 1. CONSTANTS & CONFIGURATION
# ============================================================================

DEFAULT_MODEL_ID = "gemini-3.8-flash"
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

LEVEL_OPTIONS = [
    "Complete Beginner (Zero Prior Context)",
    "Junior / Foundational (Knows Basics)",
    "Intermediate (Building Production Skills)",
    "Advanced / Senior (Deep Internals & Architecture)",
]

LEARNING_STYLE_OPTIONS = [
    "Project-First (Build & Ship Everything)",
    "First-Principles & Theory-Heavy (Academic & Mathematical Rigor)",
    "Balanced Engineering (50% Concepts, 50% Applied Labs)",
    "Fast-Track Sprint (Pareto 80/20 High-Leverage Essentials)",
]

RESOURCE_TYPE_BADGES: Dict[str, Tuple[str, str]] = {
    "documentation": ("DOC", "#38BDF8"),
    "book": ("BOOK", "#F59E0B"),
    "course": ("COURSE", "#A855F7"),
    "paper": ("PAPER", "#EC4899"),
    "video": ("VIDEO", "#EF4444"),
    "interactive": ("LAB", "#10B981"),
    "repo": ("REPO", "#E2E8F0"),
}


# ============================================================================
# 2. DOMAIN MODELS & PYDANTIC STRUCTURED OUTPUT SCHEMAS
# ============================================================================


class ResourceSchema(BaseModel):
    """Schema for a curated learning resource."""

    title: str = Field(..., description="Specific title of the book, documentation, course, or repo.")
    category: str = Field(
        ...,
        description="One of: documentation, book, course, paper, video, interactive, repo",
    )
    author_or_source: str = Field(..., description="Author, platform, or official organization.")
    url: str = Field(..., description="Direct canonical URL (or official search/domain URL).")
    why_essential: str = Field(
        ...,
        description="1-sentence explanation of why this specific resource is high-signal.",
    )


class CapstoneRubricSchema(BaseModel):
    """Schema for a hands-on phase milestone project."""

    title: str = Field(..., description="Concrete engineering or practical deliverable title.")
    problem_statement: str = Field(
        ...,
        description="Real-world specification of what the learner must build or solve.",
    )
    technical_requirements: List[str] = Field(
        ...,
        description="3 to 4 concrete acceptance criteria or technical constraints.",
    )
    verification_test: str = Field(
        ...,
        description="How the learner can objectively test or benchmark their finished work.",
    )


class RoadmapNodeSchema(BaseModel):
    """Schema for an individual learning phase/module in the DAG."""

    node_id: str = Field(..., description="Short identifier like 'M01', 'M02', 'M03'.")
    phase_number: int = Field(..., description="1-indexed sequential order.")
    title: str = Field(..., description="Action-oriented module title.")
    subtitle: str = Field(..., description="Concise architectural focus or theme.")
    estimated_hours: int = Field(..., description="Realistic hours needed to master this phase.")
    difficulty_tier: str = Field(
        ...,
        description="One of: Foundation, Core, Advanced, Mastery",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="List of previous node_ids required before starting this node (e.g. ['M01']).",
    )
    core_concepts: List[str] = Field(
        ...,
        description="4 to 6 specific, granular technical sub-topics to master.",
    )
    mental_models: List[str] = Field(
        ...,
        description="2 to 3 key mental models or architectural intuitions learners must internalize.",
    )
    common_pitfalls: List[str] = Field(
        ...,
        description="2 specific anti-patterns or traps beginners fall into during this phase.",
    )
    capstone: CapstoneRubricSchema = Field(
        ...,
        description="Hands-on milestone project to prove mastery of this phase.",
    )
    resources: List[ResourceSchema] = Field(
        ...,
        description="2 to 4 high-signal, non-generic learning resources.",
    )


class RoadmapResponseSchema(BaseModel):
    """Top-level structured schema returned by Gemini 3.8 Flash."""

    subject: str = Field(..., description="Refined title of the learning subject.")
    tagline: str = Field(..., description="Punchy 1-line engineering summary of the curriculum.")
    target_persona: str = Field(..., description="Who this roadmap is calibrated for.")
    total_estimated_hours: int = Field(..., description="Sum of hours across all modules.")
    recommended_weekly_hours: int = Field(..., description="Target weekly commitment.")
    estimated_weeks: int = Field(..., description="Total duration in weeks.")
    executive_summary: str = Field(
        ...,
        description="2-3 sentence strategic philosophy explaining why this roadmap is ordered this way.",
    )
    core_pillars: List[str] = Field(
        ...,
        description="3 to 5 overarching competency pillars covered in this roadmap.",
    )
    nodes: List[RoadmapNodeSchema] = Field(
        ...,
        description="Ordered list of 5 to 8 comprehensive learning modules forming a DAG.",
    )


@dataclass
class GenerationRequest:
    """Encapsulates user parameters for generating a roadmap."""

    subject: str
    current_level: str
    learning_style: str
    weekly_hours: int
    target_weeks: int
    specific_focus: str = ""
    model_id: str = DEFAULT_MODEL_ID
    temperature: float = 0.3

    def validate(self) -> Tuple[bool, str]:
        """Validates user input constraints prior to API dispatch."""
        cleaned = self.subject.strip()
        if not cleaned:
            return False, "Subject cannot be empty. Specify a skill, domain, or technology."
        if len(cleaned) < 2:
            return False, "Subject must be at least 2 characters long."
        if len(cleaned) > 160:
            return False, "Subject must be under 160 characters."
        if not (1 <= self.weekly_hours <= 80):
            return False, "Weekly hours must be between 1 and 80."
        if not (1 <= self.target_weeks <= 104):
            return False, "Target duration must be between 1 and 104 weeks."
        return True, ""


# ============================================================================
# 3. PROMPT ENGINEERING & JSON SANITIZATION
# ============================================================================


def build_system_instruction() -> str:
    """Returns the system instruction for Gemini 3.8 Flash."""
    return (
        "You are a Principal Curriculum Architect and Staff Engineer. "
        "Your mission is to design rigorous, non-generic, production-grade learning roadmaps "
        "for mastering any technical, scientific, or creative discipline.\n\n"
        "CRITICAL QUALITY MANDATES:\n"
        "1. NO GENERIC SLOP: Never output vague topics like 'Learn Basics' or 'Advanced Topics'. "
        "Name exact algorithms, protocols, APIs, theorems, design patterns, and tooling.\n"
        "2. TRUE DEPENDENCY GRAPH (DAG): Each module (`node_id` M01, M02, ...) must specify realistic "
        "`prerequisites` referencing earlier `node_id`s so the learner sees how concepts compound.\n"
        "3. PROOF-OF-WORK CAPSTONES: Every module must culminate in a concrete, testable project "
        "with measurable technical requirements and an objective verification test.\n"
        "4. MENTAL MODELS & ANTI-PATTERNS: Highlight the deep intuitions that separate seniors from "
        "beginners, alongside common traps to avoid.\n"
        "5. HIGH-SIGNAL RESOURCES: Recommend authoritative books, RFCs, official docs, seminal papers, "
        "or respected courses with real canonical URLs.\n"
        "6. STRICT JSON OUTPUT: Respond ONLY with valid JSON matching the requested schema."
    )


def build_user_prompt(req: GenerationRequest) -> str:
    """Constructs the structured user prompt from a GenerationRequest."""
    total_budget_hours = req.weekly_hours * req.target_weeks
    focus_clause = (
        f"\n- Specific Focus / Constraints: {req.specific_focus.strip()}"
        if req.specific_focus.strip()
        else "\n- Specific Focus / Constraints: Comprehensive end-to-end mastery"
    )
    return (
        f"Design a complete, rigorous learning roadmap for the following specification:\n"
        f"- Target Subject: {req.subject.strip()}\n"
        f"- Current Proficiency: {req.current_level}\n"
        f"- Pedagogical Style: {req.learning_style}\n"
        f"- Time Commitment: {req.weekly_hours} hours/week over {req.target_weeks} weeks "
        f"(~{total_budget_hours} total hours budget){focus_clause}\n\n"
        f"Structure the roadmap into 5 to 7 sequential yet branching modules (`M01` through `M07`) "
        f"whose `estimated_hours` sum close to {total_budget_hours} hours. "
        f"Ensure `prerequisites` strictly reference valid prior `node_id`s."
    )


def extract_and_parse_roadmap_json(raw_text: str) -> Dict[str, Any]:
    """
    Safely extracts and parses JSON from model output, stripping markdown fences
    or trailing prose if present, and validates against `RoadmapResponseSchema`.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response payload.")

    cleaned = raw_text.strip()

    # Strip ```json ... ``` fences if present
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Fallback: locate first '{' and last '}'
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]

    parsed_dict = json.loads(cleaned)
    # Validate and normalize via Pydantic (compatible with both Pydantic v1 and v2)
    if hasattr(RoadmapResponseSchema, "model_validate"):
        validated = RoadmapResponseSchema.model_validate(parsed_dict)
        return validated.model_dump()
    validated = RoadmapResponseSchema.parse_obj(parsed_dict)
    return json.loads(validated.json())


def normalize_roadmap_dag(roadmap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures DAG integrity:
    - Guarantees unique, ordered `node_id`s (`M01`, `M02`, ...).
    - Removes self-referential or forward/non-existent prerequisite references.
    - Recomputes `total_estimated_hours` and `estimated_weeks` accurately.
    """
    nodes = roadmap.get("nodes", [])
    valid_prior_ids: set[str] = set()

    total_hours = 0
    for idx, node in enumerate(nodes, start=1):
        if not node.get("node_id"):
            node["node_id"] = f"M{idx:02d}"
        node["phase_number"] = idx

        # Filter prerequisites so only valid earlier nodes are linked
        raw_prereqs = node.get("prerequisites", []) or []
        cleaned_prereqs = [p for p in raw_prereqs if p in valid_prior_ids and p != node["node_id"]]
        if idx > 1 and not cleaned_prereqs and f"M{idx-1:02d}" in valid_prior_ids:
            cleaned_prereqs = [f"M{idx-1:02d}"]
        node["prerequisites"] = cleaned_prereqs

        valid_prior_ids.add(node["node_id"])
        hours = max(1, int(node.get("estimated_hours", 10)))
        node["estimated_hours"] = hours
        total_hours += hours

    roadmap["total_estimated_hours"] = total_hours
    weekly = max(1, int(roadmap.get("recommended_weekly_hours", 10)))
    roadmap["recommended_weekly_hours"] = weekly
    roadmap["estimated_weeks"] = max(1, round(total_hours / weekly))
    return roadmap


# ============================================================================
# 4. GEMINI 3.8 FLASH AI SERVICE LAYER
# ============================================================================


class GeminiRoadmapService:
    """
    Service wrapper around Google's official `google-genai` SDK targeting `gemini-3.8-flash`.
    Supports dependency injection (`client_factory`) for deterministic unit testing.
    """

    def __init__(
        self,
        api_key: str,
        client_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self._client_factory = client_factory

    def _create_client(self) -> Any:
        if not self.api_key:
            raise ValueError("Missing Gemini API Key. Provide a valid key in the sidebar or environment.")
        if self._client_factory is not None:
            return self._client_factory(self.api_key)

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The `google-genai` package is required. Install via `pip install google-genai`."
            ) from exc

        return genai.Client(api_key=self.api_key)

    def generate_roadmap(self, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generates and validates a structured learning roadmap using `gemini-3.8-flash`.
        Automatically falls back to JSON schema prompt enforcement if needed.
        """
        is_valid, error_msg = request.validate()
        if not is_valid:
            raise ValueError(error_msg)

        client = self._create_client()
        system_instruction = build_system_instruction()
        user_prompt = build_user_prompt(request)

        # Build generation config using google.genai.types if available, else dict config
        config: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": request.temperature,
            "response_mime_type": "application/json",
            "response_schema": RoadmapResponseSchema,
        }

        models_to_try = [request.model_id] + [
            m for m in FALLBACK_MODELS if m != request.model_id
        ]

        last_exception: Optional[Exception] = None
        for model_name in models_to_try:
            try:
                logger.info("Dispatching roadmap synthesis to model=%s", model_name)
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
                raw_text = getattr(response, "text", None)
                if not raw_text and hasattr(response, "candidates") and response.candidates:
                    raw_text = response.candidates[0].content.parts[0].text

                roadmap_dict = extract_and_parse_roadmap_json(raw_text or "")
                roadmap_dict["recommended_weekly_hours"] = request.weekly_hours
                roadmap_dict["_meta"] = {
                    "model_used": model_name,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                    "learning_style": request.learning_style,
                    "proficiency_level": request.current_level,
                }
                return normalize_roadmap_dag(roadmap_dict)

            except Exception as exc:
                logger.warning("Model %s attempt failed: %s", model_name, exc)
                last_exception = exc
                # If failure is strictly an authentication or quota error, do not silently mask it
                err_str = str(exc).lower()
                if "api_key_invalid" in err_str or "invalid api key" in err_str or "permission_denied" in err_str:
                    break

        raise RuntimeError(
            f"Roadmap synthesis failed ({type(last_exception).__name__}): {last_exception}"
        ) from last_exception


# ============================================================================
# 5. BUILT-IN ARCHITECTURAL BLUEPRINTS (DEMO / OFFLINE INSPECTION)
# ============================================================================


def get_sample_roadmap() -> Dict[str, Any]:
    """
    Provides a rich, pre-computed sample roadmap ("Distributed Systems & Consensus Engineering")
    so users can immediately test the interactive DAG, progress tracking, and exporters
    even before entering a Gemini API key.
    """
    sample = {
        "subject": "Distributed Systems & Consensus Engineering",
        "tagline": "From physical network framing and RPC semantics to Raft replication, MVCC storage engines, and Geo-Distributed Spanner architectures.",
        "target_persona": "Intermediate Backend Engineer transitioning to Distributed Infrastructure",
        "total_estimated_hours": 120,
        "recommended_weekly_hours": 12,
        "estimated_weeks": 10,
        "executive_summary": (
            "Distributed systems fail in subtle, non-deterministic ways caused by partial network partitions "
            "and unsynchronized physical clocks. This curriculum rejects superficial cloud tutorials in favor "
            "of first-principles failure modeling: you will first master time, ordering, and replication primitives, "
            "then implement fault-tolerant consensus and distributed transactions from scratch."
        ),
        "core_pillars": [
            "Time, Clocks & Causal Ordering",
            "Fault-Tolerant State Machine Replication (Raft/Paxos)",
            "Partitioning, Consistent Hashing & Gossip",
            "Distributed Transactions, 2PC & MVCC Isolation",
        ],
        "_meta": {
            "model_used": "gemini-3.8-flash",
            "generated_at": "Blueprint Preset // Ready",
            "learning_style": "Project-First (Build & Ship Everything)",
            "proficiency_level": "Intermediate (Building Production Skills)",
        },
        "nodes": [
            {
                "node_id": "M01",
                "phase_number": 1,
                "title": "Network Failure Models, RPC Semantics & Framing",
                "subtitle": "Partial failures, idempotency, backpressure, and wire protocols",
                "estimated_hours": 18,
                "difficulty_tier": "Foundation",
                "prerequisites": [],
                "core_concepts": [
                    "Two Generals & FLP Impossibility Intuition",
                    "At-least-once vs. At-most-once vs. Exactly-once RPC via Idempotency Keys",
                    "TCP Head-of-Line Blocking, Keepalives, and Epoll/Kqueue Event Loops",
                    "Tail Latency Amplification & Hedged Requests",
                    "Bounded Queues, Token Bucket Rate Limiting & Circuit Breakers",
                ],
                "mental_models": [
                    "A timeout never tells you whether the remote node executed the mutation—only that the reply was lost or delayed.",
                    "Unbounded queues convert transient latency spikes into catastrophic Out-Of-Memory cascades.",
                ],
                "common_pitfalls": [
                    "Retrying non-idempotent mutations blindly without jittered exponential backoff.",
                    "Treating local loopback network reliability as representative of cross-AZ WAN links.",
                ],
                "capstone": {
                    "title": "Fault-Injecting Multiplexed RPC Framework",
                    "problem_statement": "Build a binary framed RPC client/server supporting request cancellation, deadlines, jittered retries, and deterministic packet drop/reorder injection.",
                    "technical_requirements": [
                        "Implement length-prefixed framing with CRC32 checksum verification.",
                        "Enforce per-request deadline propagation and server-side cancellation.",
                        "Pass a chaos test suite dropping 15% of packets and delaying 10% by 300ms.",
                    ],
                    "verification_test": "Run 10,000 concurrent idempotent mutations under simulated 20% packet loss with zero duplicate state applications.",
                },
                "resources": [
                    {
                        "title": "Designing Data-Intensive Applications (Ch. 1, 4, 8)",
                        "category": "book",
                        "author_or_source": "Martin Kleppmann (O'Reilly)",
                        "url": "https://dataintensive.net/",
                        "why_essential": "Definitive analysis of partial failures, unreliable networks, and process pauses.",
                    },
                    {
                        "title": "The Tail at Scale",
                        "category": "paper",
                        "author_or_source": "Jeffrey Dean & Luiz André Barroso (Google)",
                        "url": "https://cacm.acm.org/research/the-tail-at-scale/",
                        "why_essential": "Explains why 99th-percentile latency dominates large fan-out services and how hedging mitigates it.",
                    },
                ],
            },
            {
                "node_id": "M02",
                "phase_number": 2,
                "title": "Logical Time, Causal Broadcast & CRDTs",
                "subtitle": "Ordering events without synchronized physical clocks",
                "estimated_hours": 20,
                "difficulty_tier": "Foundation",
                "prerequisites": ["M01"],
                "core_concepts": [
                    "Quartz Clock Drift, NTP Skew & Monotonic vs. Wall-Clock Time",
                    "Happens-Before Relation (→) & Lamport Logical Timestamps",
                    "Vector Clocks & Concurrent Conflict Detection",
                    "Hybrid Logical Clocks (HLC) in CockroachDB/MongoDB",
                    "State-based (CvRDT) & Operation-based (CmRDT) Conflict-Free Replicated Data Types",
                ],
                "mental_models": [
                    "Physical time is an illusion across machines; causality is tracked via message exchange history.",
                    "Vector clocks detect concurrency; CRDT semilattices resolve concurrency deterministically without coordination.",
                ],
                "common_pitfalls": [
                    "Using Last-Write-Wins (LWW) with unsynchronized `System.currentTimeMillis()` wall clocks.",
                    "Allowing vector clock metadata to grow unboundedly with ephemeral client IDs.",
                ],
                "capstone": {
                    "title": "Peer-to-Peer Collaborative JSON Document Store via HLC & LWW-Element-Set",
                    "problem_statement": "Construct a 5-node mesh state store that accepts concurrent writes during a complete network split and converges identically upon partition heal.",
                    "technical_requirements": [
                        "Implement a Hybrid Logical Clock generator resistant to 500ms NTP backward jumps.",
                        "Support add/remove dictionary keys and set elements with tombstone garbage collection.",
                        "Verify strong eventual consistency across randomized network partition schedules.",
                    ],
                    "verification_test": "Property-based test generating 5,000 random interleaved operations across 5 partitioned replicas; assert byte-identical state after anti-entropy sync.",
                },
                "resources": [
                    {
                        "title": "Time, Clocks, and the Ordering of Events in a Distributed System",
                        "category": "paper",
                        "author_or_source": "Leslie Lamport (1978)",
                        "url": "https://lamport.azurewebsites.net/pubs/time-clocks.pdf",
                        "why_essential": "The foundational paper of distributed computing introducing partial ordering and logical clocks.",
                    },
                    {
                        "title": "Logical Physical Clocks and Consistent Snapshots (HLC)",
                        "category": "paper",
                        "author_or_source": "Kulkarni et al.",
                        "url": "https://cse.buffalo.edu/tech-reports/2014-04.pdf",
                        "why_essential": "Shows how to combine physical NTP time with Lamport causality in 64-bit timestamps.",
                    },
                ],
            },
            {
                "node_id": "M03",
                "phase_number": 3,
                "title": "Replicated Log Consensus & Raft Internals",
                "subtitle": "Leader election, log safety, linearizable reads, and snapshotting",
                "estimated_hours": 32,
                "difficulty_tier": "Core",
                "prerequisites": ["M01", "M02"],
                "core_concepts": [
                    "State Machine Replication & Linearizability (Herlihy & Wing)",
                    "Raft Leader Election, Split-Vote Randomization & Term Monotonicity",
                    "Log Matching Property, Election Safety & Commitment Rules (Current-Term Entry)",
                    "Log Compaction, InstallSnapshot RPC & Copy-on-Write State Serialization",
                    "ReadIndex & LeaseRead Optimizations for Linearizable Reads without Disk Quorum",
                ],
                "mental_models": [
                    "A leader must NEVER commit log entries from an older term by counting replicas alone; it must commit an entry from its own term first (Figure 8 rule).",
                    "Linearizability is a local property of single objects; serializability is a global property of multi-object transactions.",
                ],
                "common_pitfalls": [
                    "Serving stale reads from a deposed leader partitioned from the majority quorum.",
                    "Forgetting to fsync `currentTerm` and `votedFor` to disk before replying to `RequestVote`.",
                ],
                "capstone": {
                    "title": "Linearizable Raft Key-Value Store (MIT 6.5840 Grade)",
                    "problem_statement": "Implement the complete Raft consensus algorithm with persistent write-ahead logging, dynamic log compaction, and linearizable Put/Append/Get semantics.",
                    "technical_requirements": [
                        "Implement Leader Election, AppendEntries heartbeats, and fast conflict backup optimization.",
                        "Persist state across sudden node crashes and restarts using atomic write-ahead logs.",
                        "Pass Porcupine linearizability verification under concurrent client traffic and network partitions.",
                    ],
                    "verification_test": "Execute 200 iterations of randomized network partition + crash-recovery tests verified against the Porcupine linearizability checker.",
                },
                "resources": [
                    {
                        "title": "In Search of an Understandable Consensus Algorithm (Extended Raft)",
                        "category": "paper",
                        "author_or_source": "Diego Ongaro & John Ousterhout",
                        "url": "https://raft.github.io/raft.pdf",
                        "why_essential": "Contains Figure 2 and Figure 8—the exact specification required to implement Raft correctly.",
                    },
                    {
                        "title": "MIT 6.5840 Distributed Systems Labs",
                        "category": "course",
                        "author_or_source": "Robert Morris & Frans Kaashoek (MIT)",
                        "url": "https://pdos.csail.mit.edu/6.824/",
                        "why_essential": "Industry gold-standard laboratory sequence for building fault-tolerant Raft systems.",
                    },
                ],
            },
            {
                "node_id": "M04",
                "phase_number": 4,
                "title": "Data Partitioning, Dynamo Quorums & Anti-Entropy",
                "subtitle": "Horizontal scaling, consistent hashing, and sloppy quorums",
                "estimated_hours": 22,
                "difficulty_tier": "Core",
                "prerequisites": ["M02"],
                "core_concepts": [
                    "Consistent Hashing Rings with Virtual Nodes (vnodes) & Bounded Load",
                    "Leaderless Replication: Strict Quorums (R + W > N) vs. Sloppy Quorums",
                    "Hinted Handoff, Read Repair & Merkle Tree Anti-Entropy Synchronization",
                    "SWIM Gossip Membership Protocol & Phi Accrual Failure Detectors",
                    "Range Partitioning vs. Hash Partitioning Hot-Spot Mitigation",
                ],
                "mental_models": [
                    "Even with R + W > N, strict quorum reads are NOT linearizable without read-repair synchronization on concurrent writes.",
                    "Merkle trees reduce cross-replica synchronization bandwidth from O(Data Size) to O(Log(Key Ranges)).",
                ],
                "common_pitfalls": [
                    "Using naive modulo hashing (`hash(key) % N`) which reshuffles ~100% of keys on node join.",
                    "Triggering full cluster rebalancing storms during transient 30-second node restarts.",
                ],
                "capstone": {
                    "title": "Dynamo-Style Partitioned Storage Ring with Merkle Sync",
                    "problem_statement": "Build a leaderless horizontally partitioned storage cluster using virtual-node consistent hashing, tunable (N, R, W) quorums, and background Merkle tree repair.",
                    "technical_requirements": [
                        "Distribute keys across 128 virtual nodes per physical server with <8% key movement on scale-out.",
                        "Implement Merkle tree diffing to heal diverging key ranges after node recovery.",
                        "Implement SWIM gossip failure detection to route around dead nodes within 3 seconds.",
                    ],
                    "verification_test": "Kill 2 of 5 storage nodes during a 50k write load, bring them back blank, and verify automatic Merkle healing within 10 seconds.",
                },
                "resources": [
                    {
                        "title": "Dynamo: Amazon's Highly Available Key-value Store",
                        "category": "paper",
                        "author_or_source": "DeCandia et al. (SOSP 2007)",
                        "url": "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf",
                        "why_essential": "Seminal architecture combining consistent hashing, sloppy quorums, vector clocks, and Merkle trees.",
                    },
                    {
                        "title": "Architecture of Apache Cassandra / ScyllaDB Internals",
                        "category": "documentation",
                        "author_or_source": "ScyllaDB University",
                        "url": "https://docs.scylladb.com/stable/architecture/",
                        "why_essential": "Production reference for vnodes, ring topology, and gossip anti-entropy.",
                    },
                ],
            },
            {
                "node_id": "M05",
                "phase_number": 5,
                "title": "Distributed Transactions, MVCC & Geo-Scale Spanner",
                "subtitle": "Two-Phase Commit, Snapshot Isolation, Percolator, and TrueTime",
                "estimated_hours": 28,
                "difficulty_tier": "Mastery",
                "prerequisites": ["M03", "M04"],
                "core_concepts": [
                    "Multi-Version Concurrency Control (MVCC), Write Skew & Serializable Snapshot Isolation (SSI)",
                    "Atomic Commitment: Two-Phase Commit (2PC) over Raft Consensus Groups",
                    "Google Percolator: Distributed Snapshot Isolation via Primary Lock Pointers",
                    "Google Spanner Architecture: TrueTime Bounded Uncertainty [earliest, latest] & Commit Wait",
                    "Deterministic Databases (Calvin) vs. Loosely Synchronized Multi-Shard Engines",
                ],
                "mental_models": [
                    "2PC is an atomic commitment protocol, not a availability protocol; pairing 2PC participants with Raft groups eliminates single-node coordinator blocking.",
                    "Spanner converts physical clock uncertainty ε into a deterministic commit wait delay of 2ε to guarantee external consistency.",
                ],
                "common_pitfalls": [
                    "Assuming Snapshot Isolation prevents Write Skew anomalies across disjoint keys.",
                    "Holding pessimistic locks across slow wide-area network round-trips without deadlock detection (Wait-Die / Wound-Wait).",
                ],
                "capstone": {
                    "title": "Sharded Multi-Raft Transactional KV Engine (Percolator Model)",
                    "problem_statement": "Combine your Raft state machine (M03) and partitioning layer (M04) to support ACID cross-shard transactions using Percolator-style two-phase locking and MVCC timestamps.",
                    "technical_requirements": [
                        "Support multi-key `Begin`, `Get`, `Set`, and `Commit` across 3 independent Raft shard groups.",
                        "Implement primary-lock atomicity and asynchronous secondary lock cleanup on coordinator crash.",
                        "Pass the Jepsen-style bank-transfer invariant test (constant total balance across 100 accounts).",
                    ],
                    "verification_test": "Run 50 concurrent cross-shard bank transfer workers while randomly killing shard leaders; verify zero lost updates, zero dirty reads, and exact sum preservation.",
                },
                "resources": [
                    {
                        "title": "Spanner: Google's Globally-Distributed Database",
                        "category": "paper",
                        "author_or_source": "Corbett et al. (OSDI 2012)",
                        "url": "https://research.google/pubs/spanner-googles-globally-distributed-database/",
                        "why_essential": "Explains how Paxos/Raft shard groups, 2PC, MVCC, and TrueTime unify into externally consistent SQL.",
                    },
                    {
                        "title": "Large-scale Incremental Processing Using Distributed Transactions (Percolator)",
                        "category": "paper",
                        "author_or_source": "Peng & Dabek (Google)",
                        "url": "https://research.google/pubs/large-scale-incremental-processing-using-distributed-transactions-and-notifications/",
                        "why_essential": "Practical blueprint for implementing distributed ACID transactions atop a key-value store (used by TiKV).",
                    },
                ],
            },
        ],
    }
    return normalize_roadmap_dag(sample)


# ============================================================================
# 6. EXPORT & VISUALIZATION GENERATORS (SVG DAG, MARKDOWN, GRAPHVIZ)
# ============================================================================


def compute_progress_metrics(
    roadmap: Dict[str, Any], completed_ids: set[str]
) -> Dict[str, Any]:
    """Calculates module and hour completion metrics for a roadmap."""
    nodes = roadmap.get("nodes", [])
    total_nodes = len(nodes)
    if total_nodes == 0:
        return {
            "completed_nodes": 0,
            "total_nodes": 0,
            "completed_hours": 0,
            "total_hours": 0,
            "percent_complete": 0.0,
            "unlocked_nodes": [],
        }

    total_hours = sum(int(n.get("estimated_hours", 0)) for n in nodes)
    completed_hours = sum(
        int(n.get("estimated_hours", 0))
        for n in nodes
        if n.get("node_id") in completed_ids
    )
    completed_count = sum(1 for n in nodes if n.get("node_id") in completed_ids)
    percent = round((completed_hours / total_hours) * 100, 1) if total_hours > 0 else 0.0

    # Determine which nodes are currently unlocked (all prerequisites satisfied, not yet completed)
    unlocked_ids: List[str] = []
    for node in nodes:
        nid = node.get("node_id", "")
        if nid in completed_ids:
            continue
        prereqs = node.get("prerequisites", []) or []
        if all(p in completed_ids for p in prereqs):
            unlocked_ids.append(nid)

    return {
        "completed_nodes": completed_count,
        "total_nodes": total_nodes,
        "completed_hours": completed_hours,
        "total_hours": total_hours,
        "percent_complete": percent,
        "unlocked_nodes": unlocked_ids,
    }


def render_svg_topology_graph(
    roadmap: Dict[str, Any],
    completed_ids: set[str],
    selected_node_id: Optional[str] = None,
) -> str:
    """
    Generates a clean, responsive dark-themed SVG blueprint dependency topology diagram
    showing every node, its status (Completed / Ready / Locked), estimated hours, and edges.
    """
    nodes = roadmap.get("nodes", [])
    if not nodes:
        return ""

    metrics = compute_progress_metrics(roadmap, completed_ids)
    unlocked_set = set(metrics["unlocked_nodes"])

    # Assign topological layers based on longest prerequisite path
    node_map = {n["node_id"]: n for n in nodes}
    layers: Dict[str, int] = {}

    def get_layer(nid: str, visited: Optional[set[str]] = None) -> int:
        if nid in layers:
            return layers[nid]
        if visited is None:
            visited = set()
        if nid in visited or nid not in node_map:
            return 0
        visited.add(nid)
        prereqs = node_map[nid].get("prerequisites", []) or []
        if not prereqs:
            layers[nid] = 0
        else:
            layers[nid] = 1 + max(get_layer(p, visited.copy()) for p in prereqs)
        return layers[nid]

    for n in nodes:
        get_layer(n["node_id"])

    # Group nodes by layer
    layer_groups: Dict[int, List[Dict[str, Any]]] = {}
    for n in nodes:
        l_idx = layers.get(n["node_id"], 0)
        layer_groups.setdefault(l_idx, []).append(n)

    max_layer = max(layer_groups.keys()) if layer_groups else 0
    max_per_layer = max(len(g) for g in layer_groups.values()) if layer_groups else 1

    # SVG Canvas Geometry
    card_w = 215
    card_h = 84
    col_gap = 68
    row_gap = 32
    pad_x = 36
    pad_y = 34

    svg_width = max(780, pad_x * 2 + (max_layer + 1) * card_w + max_layer * col_gap)
    svg_height = max(210, pad_y * 2 + max_per_layer * card_h + (max_per_layer - 1) * row_gap)

    coords: Dict[str, Tuple[float, float]] = {}
    for l_idx in range(max_layer + 1):
        group = layer_groups.get(l_idx, [])
        col_x = pad_x + l_idx * (card_w + col_gap)
        total_col_h = len(group) * card_h + max(0, len(group) - 1) * row_gap
        start_y = (svg_height - total_col_h) / 2.0
        for row_idx, n in enumerate(group):
            row_y = start_y + row_idx * (card_h + row_gap)
            coords[n["node_id"]] = (col_x, row_y)

    # Render Edges
    edge_elements: List[str] = []
    for n in nodes:
        dst_id = n["node_id"]
        if dst_id not in coords:
            continue
        dst_x, dst_y = coords[dst_id]
        dst_cy = dst_y + card_h / 2.0
        for src_id in n.get("prerequisites", []) or []:
            if src_id not in coords:
                continue
            src_x, src_y = coords[src_id]
            src_right = src_x + card_w
            src_cy = src_y + card_h / 2.0
            mid_x = (src_right + dst_x) / 2.0

            is_active_edge = src_id in completed_ids
            stroke_color = "#10B981" if is_active_edge else "#F59E0B" if dst_id in unlocked_set else "#334155"
            stroke_dash = "" if is_active_edge else 'stroke-dasharray="5,4"'
            opacity = "0.95" if (is_active_edge or dst_id in unlocked_set) else "0.55"

            path_d = f"M {src_right:.1f} {src_cy:.1f} C {mid_x:.1f} {src_cy:.1f}, {mid_x:.1f} {dst_cy:.1f}, {dst_x:.1f} {dst_cy:.1f}"
            edge_elements.append(
                f'<path d="{path_d}" fill="none" stroke="{stroke_color}" '
                f'stroke-width="2.2" {stroke_dash} opacity="{opacity}" marker-end="url(#arrow-{stroke_color.lstrip("#")})"/>'
            )

    # Render Nodes
    node_elements: List[str] = []
    for n in nodes:
        nid = n["node_id"]
        x, y = coords[nid]
        title = n.get("title", "")
        short_title = (title[:24] + "…") if len(title) > 25 else title
        hours = n.get("estimated_hours", 0)
        tier = n.get("difficulty_tier", "Core").upper()

        if nid in completed_ids:
            border_col = "#10B981"
            bg_col = "#06281E"
            badge_text = "✓ COMPLETED"
            badge_col = "#10B981"
        elif nid in unlocked_set:
            border_col = "#F59E0B"
            bg_col = "#1E1607"
            badge_text = "◉ ACTIVE / READY"
            badge_col = "#F59E0B"
        else:
            border_col = "#334155"
            bg_col = "#0F141C"
            badge_text = "⊘ LOCKED"
            badge_col = "#64748B"

        if selected_node_id == nid:
            stroke_width = "2.8"
            glow_filter = 'filter="url(#glow)"'
        else:
            stroke_width = "1.5"
            glow_filter = ""

        safe_title = html.escape(short_title)
        safe_id = html.escape(nid)
        safe_tier = html.escape(tier[:10])

        node_elements.append(
            f'<g transform="translate({x:.1f},{y:.1f})" {glow_filter}>'
            f'  <rect width="{card_w}" height="{card_h}" rx="6" fill="{bg_col}" '
            f'        stroke="{border_col}" stroke-width="{stroke_width}"/>'
            f'  <text x="12" y="21" fill="{badge_col}" font-family="JetBrains Mono, monospace" '
            f'        font-size="10" font-weight="700">{safe_id} // {badge_text}</text>'
            f'  <text x="12" y="45" fill="#F8FAFC" font-family="Space Grotesk, sans-serif" '
            f'        font-size="12.5" font-weight="600">{safe_title}</text>'
            f'  <text x="12" y="68" fill="#94A3B8" font-family="JetBrains Mono, monospace" '
            f'        font-size="10">{hours}h  •  {safe_tier}</text>'
            f"</g>"
        )

    svg = f"""
    <div style="width:100%; overflow-x:auto; background:#090C10; border:1px solid #1E293B; border-radius:8px; padding:8px;">
      <svg viewBox="0 0 {svg_width} {svg_height}" width="100%" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="blueprint-grid" width="24" height="24" patternUnits="userSpaceOnUse">
            <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#141C26" stroke-width="1"/>
          </pattern>
          <filter id="glow" x="-15%" y="-15%" width="130%" height="130%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#F59E0B" flood-opacity="0.45"/>
          </filter>
          <marker id="arrow-10B981" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981"/>
          </marker>
          <marker id="arrow-F59E0B" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#F59E0B"/>
          </marker>
          <marker id="arrow-334155" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/>
          </marker>
        </defs>
        <rect width="100%" height="100%" fill="url(#blueprint-grid)"/>
        {''.join(edge_elements)}
        {''.join(node_elements)}
      </svg>
    </div>
    """
    return svg.strip()


def export_roadmap_to_markdown(roadmap: Dict[str, Any], completed_ids: Optional[set[str]] = None) -> str:
    """Exports the complete roadmap into a clean, portable Markdown study specification."""
    completed = completed_ids or set()
    lines: List[str] = [
        f"# {roadmap.get('subject', 'Learning Roadmap')}",
        f"> **{roadmap.get('tagline', '')}**",
        "",
        "## Architectural Telemetry",
        f"- **Target Persona**: {roadmap.get('target_persona', 'Learner')}",
        f"- **Total Time Budget**: {roadmap.get('total_estimated_hours', 0)} hours "
        f"(~{roadmap.get('estimated_weeks', 0)} weeks @ {roadmap.get('recommended_weekly_hours', 0)} hrs/week)",
        f"- **Core Competency Pillars**: {', '.join(roadmap.get('core_pillars', []))}",
        "",
        "## Executive Strategy",
        roadmap.get("executive_summary", ""),
        "",
        "## Curriculum Modules (DAG)",
    ]

    for node in roadmap.get("nodes", []):
        nid = node.get("node_id", "M00")
        check = "x" if nid in completed else " "
        prereqs = ", ".join(node.get("prerequisites", [])) or "None (Entry Node)"
        lines.extend(
            [
                f"### [{check}] {nid}: {node.get('title', '')}",
                f"**{node.get('subtitle', '')}**",
                f"- **Difficulty Tier**: `{node.get('difficulty_tier', 'Core')}` | "
                f"**Estimated Effort**: `{node.get('estimated_hours', 0)} hours` | "
                f"**Prerequisites**: `{prereqs}`",
                "",
                "#### Core Technical Sub-Topics",
            ]
        )
        for concept in node.get("core_concepts", []):
            lines.append(f"- [ ] {concept}")

        lines.append("\n#### Key Mental Models")
        for model in node.get("mental_models", []):
            lines.append(f"- {model}")

        lines.append("\n#### Anti-Patterns & Common Pitfalls")
        for pitfall in node.get("common_pitfalls", []):
            lines.append(f"- ⚠️ {pitfall}")

        capstone = node.get("capstone", {})
        if capstone:
            lines.extend(
                [
                    "",
                    f"#### Proof-of-Work Capstone: {capstone.get('title', '')}",
                    f"{capstone.get('problem_statement', '')}",
                    "",
                    "**Acceptance Criteria:**",
                ]
            )
            for req in capstone.get("technical_requirements", []):
                lines.append(f"- {req}")
            lines.append(f"\n**Verification Benchmark:** {capstone.get('verification_test', '')}")

        resources = node.get("resources", [])
        if resources:
            lines.append("\n#### Curated Reference Matrix")
            for res in resources:
                lines.append(
                    f"- **[{res.get('title', 'Resource')}]({res.get('url', '#')})** "
                    f"(`{res.get('category', 'doc').upper()}` — {res.get('author_or_source', '')}): "
                    f"{res.get('why_essential', '')}"
                )
        lines.append("")

    return "\n".join(lines)


def export_roadmap_to_dot(roadmap: Dict[str, Any]) -> str:
    """Exports the roadmap dependency graph as a Graphviz DOT specification."""
    lines = [
        "digraph LearningRoadmap {",
        '  rankdir="LR";',
        '  bgcolor="#090C10";',
        '  node [shape=box, style="rounded,filled", fillcolor="#121721", color="#F59E0B", fontname="Helvetica", fontcolor="#F8FAFC"];',
        '  edge [color="#38BDF8", penwidth=1.8];',
    ]
    for node in roadmap.get("nodes", []):
        nid = node.get("node_id", "M00")
        label = f"{nid}: {node.get('title', '')}\\n({node.get('estimated_hours', 0)}h | {node.get('difficulty_tier', '')})"
        escaped_label = label.replace('"', '\\"')
        lines.append(f'  "{nid}" [label="{escaped_label}"];')
        for prereq in node.get("prerequisites", []) or []:
            lines.append(f'  "{prereq}" -> "{nid}";')
    lines.append("}")
    return "\n".join(lines)


# ============================================================================
# 7. BESPOKE DARK BRUTALIST / ARCHITECTURAL BLUEPRINT THEME (CSS)
# ============================================================================


def inject_custom_dark_css() -> None:
    """Injects custom CSS for the 'Obsidian & Amber Technical Cartographer' dark theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg-void: #090C10;
            --bg-panel: #11161F;
            --bg-card: #151C28;
            --border-subtle: #222D3D;
            --border-amber: #F59E0B;
            --accent-amber: #F59E0B;
            --accent-emerald: #10B981;
            --accent-cyan: #38BDF8;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
        }

        /* Main App Background with subtle architectural dot matrix */
        .stApp {
            background-color: var(--bg-void);
            background-image: radial-gradient(#1E293B 1px, transparent 1px);
            background-size: 24px 24px;
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0B0F17 !important;
            border-right: 1px solid var(--border-subtle);
        }

        /* Typography overrides */
        h1, h2, h3, .font-display {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--text-primary) !important;
        }

        code, pre, .font-mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Custom Header Banner */
        .blueprint-header {
            background: linear-gradient(135deg, rgba(21,28,40,0.96) 0%, rgba(11,15,23,0.98) 100%);
            border: 1px solid var(--border-subtle);
            border-left: 4px solid var(--accent-amber);
            border-radius: 8px;
            padding: 22px 26px;
            margin-bottom: 20px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        }

        .telemetry-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .telemetry-card {
            background: rgba(17, 22, 31, 0.92);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 14px 16px;
        }

        .telemetry-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .telemetry-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Phase Module Card */
        .phase-card {
            background: rgba(21, 28, 40, 0.94);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
            transition: border-color 0.15s ease;
        }

        .phase-card:hover {
            border-color: #3B4B64;
        }

        .phase-card-completed {
            border-left: 4px solid var(--accent-emerald);
        }

        .phase-card-active {
            border-left: 4px solid var(--accent-amber);
        }

        .phase-card-locked {
            border-left: 4px solid #334155;
            opacity: 0.88;
        }

        .badge-mono {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 4px;
            margin-right: 8px;
            text-transform: uppercase;
        }

        .capstone-box {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 6px;
            padding: 15px 18px;
            margin-top: 12px;
        }

        .resource-row {
            background: #0E131B;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 11px 14px;
            margin-bottom: 8px;
        }

        /* Primary Button Restyling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%) !important;
            color: #090C10 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 6px !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .stButton > button[kind="secondary"] {
            background-color: #151C28 !important;
            color: #E2E8F0 !important;
            border: 1px solid #2D3B4E !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 8. STREAMLIT UI COMPONENTS & STATE ORCHESTRATION
# ============================================================================


def init_session_state() -> None:
    """Initializes persistent Streamlit session state variables."""
    if "active_roadmap" not in st.session_state:
        st.session_state["active_roadmap"] = get_sample_roadmap()
    if "completed_nodes" not in st.session_state:
        st.session_state["completed_nodes"] = {"M01"}
    if "roadmap_history" not in st.session_state:
        st.session_state["roadmap_history"] = []
    if "selected_node_id" not in st.session_state:
        st.session_state["selected_node_id"] = "M02"


def render_sidebar() -> Tuple[str, GenerationRequest, bool, bool]:
    """Renders the control sidebar and returns API key, request config, and trigger flags."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 12px; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #F59E0B; letter-spacing: 0.1em;">
                    PATHFINDER // ENGINE v3.8
                </div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: #F8FAFC;">
                    Curriculum Architect
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                    Powered by Google <code>gemini-3.8-flash</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        env_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "GEMINI API KEY",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Google AI Studio API key (`GEMINI_API_KEY`).",
        )

        st.markdown("#### 01 // TARGET SPECIFICATION")
        subject = st.text_input(
            "Subject or Skill Domain",
            value="Rust Async Runtime & Systems Programming",
            placeholder="e.g., Quantum Error Correction, Compiler Design, Kubernetes Internals",
        )

        current_level = st.selectbox("Current Proficiency", options=LEVEL_OPTIONS, index=1)
        learning_style = st.selectbox("Pedagogical Mode", options=LEARNING_STYLE_OPTIONS, index=0)

        col_w, col_t = st.columns(2)
        with col_w:
            weekly_hours = st.number_input("Hours / Week", min_value=2, max_value=60, value=10, step=2)
        with col_t:
            target_weeks = st.number_input("Target Weeks", min_value=2, max_value=52, value=8, step=1)

        specific_focus = st.text_area(
            "Specific Focus / Constraints (Optional)",
            placeholder="e.g., Focus on Linux io_uring, zero-copy networking, and lock-free ring buffers.",
            height=78,
        )

        with st.expander("ADVANCED TELEMETRY SETTINGS"):
            model_id = st.selectbox(
                "Gemini Model Endpoint",
                options=FALLBACK_MODELS,
                index=0,
                help="Defaults to `gemini-3.8-flash`.",
            )
            temperature = st.slider(
                "Synthesis Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )

        generate_clicked = st.button(
            "⚡ SYNTHESIZE ROADMAP",
            type="primary",
            use_container_width=True,
        )

        load_demo_clicked = st.button(
            "↺ LOAD SAMPLE BLUEPRINT",
            type="secondary",
            use_container_width=True,
            help="Load the built-in Distributed Systems architectural blueprint.",
        )

        if st.session_state.get("roadmap_history"):
            st.markdown("#### 02 // SESSION ARCHIVE")
            for idx, item in enumerate(reversed(st.session_state["roadmap_history"][-5:])):
                label = f"{item.get('subject', 'Untitled')[:28]} ({item.get('total_estimated_hours', 0)}h)"
                if st.button(f"📂 {label}", key=f"hist_{idx}", use_container_width=True):
                    st.session_state["active_roadmap"] = item
                    st.session_state["completed_nodes"] = set()
                    st.rerun()

        req = GenerationRequest(
            subject=subject,
            current_level=current_level,
            learning_style=learning_style,
            weekly_hours=int(weekly_hours),
            target_weeks=int(target_weeks),
            specific_focus=specific_focus,
            model_id=model_id,
            temperature=float(temperature),
        )
        return api_key, req, generate_clicked, load_demo_clicked


def render_telemetry_header(roadmap: Dict[str, Any], completed_ids: set[str]) -> None:
    """Renders the top architectural banner and quantitative telemetry strip."""
    metrics = compute_progress_metrics(roadmap, completed_ids)
    meta = roadmap.get("_meta", {})
    model_used = meta.get("model_used", DEFAULT_MODEL_ID)
    pillars_html = "".join(
        f'<span class="badge-mono" style="background:#1E293B; color:#38BDF8; border:1px solid #334155; margin-bottom:6px;">{html.escape(p)}</span>'
        for p in roadmap.get("core_pillars", [])
    )

    st.markdown(
        f"""
        <div class="blueprint-header">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
                <span class="badge-mono" style="background:rgba(245,158,11,0.15); color:#F59E0B; border:1px solid rgba(245,158,11,0.4);">
                    BLUEPRINT // {html.escape(model_used.upper())}
                </span>
                <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#64748B;">
                    PERSONA: {html.escape(roadmap.get('target_persona', 'Technical Learner'))}
                </span>
            </div>
            <h1 style="margin:0 0 8px 0; font-size:2.05rem; line-height:1.2;">
                {html.escape(roadmap.get('subject', 'Learning Roadmap'))}
            </h1>
            <p style="color:#CBD5E1; font-size:1.0rem; margin:0 0 14px 0; line-height:1.5;">
                {html.escape(roadmap.get('tagline', ''))}
            </p>
            <div style="margin-bottom:12px;">{pillars_html}</div>
            <div style="font-size:0.88rem; color:#94A3B8; border-top:1px solid #1E293B; padding-top:10px; line-height:1.55;">
                <strong style="color:#E2E8F0; font-family:'JetBrains Mono', monospace; font-size:0.78rem;">ARCHITECTURAL RATIONALE:</strong>
                {html.escape(roadmap.get('executive_summary', ''))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_ready = ", ".join(metrics["unlocked_nodes"]) if metrics["unlocked_nodes"] else "ALL COMPLETE 🎉"
    st.markdown(
        f"""
        <div class="telemetry-strip">
            <div class="telemetry-card">
                <div class="telemetry-label">TOTAL EFFORT BUDGET</div>
                <div class="telemetry-value">{metrics['total_hours']} <span style="font-size:0.9rem; color:#94A3B8;">HRS</span></div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">ESTIMATED VELOCITY</div>
                <div class="telemetry-value">{roadmap.get('estimated_weeks', 0)} <span style="font-size:0.9rem; color:#94A3B8;">WKS @ {roadmap.get('recommended_weekly_hours', 0)}h/wk</span></div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">MASTERY PROGRESS</div>
                <div class="telemetry-value" style="color:#10B981;">{metrics['percent_complete']}% <span style="font-size:0.9rem; color:#94A3B8;">({metrics['completed_nodes']}/{metrics['total_nodes']} MODULES)</span></div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">UNLOCKED / ACTIVE NODES</div>
                <div class="telemetry-value" style="color:#F59E0B; font-family:'JetBrains Mono', monospace; font-size:1.15rem;">{html.escape(next_ready)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_cards(roadmap: Dict[str, Any], completed_ids: set[str]) -> None:
    """Renders interactive phase cards with prerequisite status, mental models, and capstones."""
    metrics = compute_progress_metrics(roadmap, completed_ids)
    unlocked_set = set(metrics["unlocked_nodes"])

    for node in roadmap.get("nodes", []):
        nid = node.get("node_id", "M00")
        is_done = nid in completed_ids
        is_unlocked = nid in unlocked_set

        if is_done:
            card_cls = "phase-card phase-card-completed"
            status_badge = '<span class="badge-mono" style="background:rgba(16,185,129,0.16); color:#10B981; border:1px solid #10B981;">✓ MASTERED</span>'
        elif is_unlocked:
            card_cls = "phase-card phase-card-active"
            status_badge = '<span class="badge-mono" style="background:rgba(245,158,11,0.18); color:#F59E0B; border:1px solid #F59E0B;">◉ READY TO START</span>'
        else:
            card_cls = "phase-card phase-card-locked"
            status_badge = '<span class="badge-mono" style="background:#1E293B; color:#64748B; border:1px solid #334155;">⊘ PREREQ PENDING</span>'

        prereqs = node.get("prerequisites", []) or []
        prereq_str = ", ".join(prereqs) if prereqs else "ROOT ENTRY"
        tier = node.get("difficulty_tier", "Core")
        hours = node.get("estimated_hours", 0)

        concepts_html = "".join(
            f'<li style="margin-bottom:5px; color:#E2E8F0;">{html.escape(c)}</li>'
            for c in node.get("core_concepts", [])
        )
        models_html = "".join(
            f'<li style="margin-bottom:5px; color:#CBD5E1;">{html.escape(m)}</li>'
            for m in node.get("mental_models", [])
        )
        pitfalls_html = "".join(
            f'<li style="margin-bottom:5px; color:#FCA5A5;">{html.escape(p)}</li>'
            for p in node.get("common_pitfalls", [])
        )

        capstone = node.get("capstone", {})
        cap_reqs_html = "".join(
            f'<li style="margin-bottom:4px; color:#E2E8F0;">{html.escape(r)}</li>'
            for r in capstone.get("technical_requirements", [])
        )

        st.markdown(
            f"""
            <div class="{card_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
                    <div>
                        <span class="badge-mono" style="background:#0F172A; color:#F8FAFC; border:1px solid #334155;">{html.escape(nid)}</span>
                        {status_badge}
                        <span class="badge-mono" style="background:#0F172A; color:#38BDF8; border:1px solid #1E293B;">{html.escape(tier)}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:#94A3B8;">
                        ⏱ {hours} HRS &nbsp;|&nbsp; DEPS: <code>{html.escape(prereq_str)}</code>
                    </div>
                </div>
                <h3 style="margin:4px 0 4px 0; font-size:1.3rem;">{html.escape(node.get('title', ''))}</h3>
                <div style="color:#94A3B8; font-size:0.9rem; margin-bottom:14px;">{html.escape(node.get('subtitle', ''))}</div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px;">
                    <div style="background:#0B0F17; border:1px solid #1E293B; border-radius:6px; padding:12px 16px;">
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#38BDF8; margin-bottom:8px;">
                            // CORE TECHNICAL CONCEPTS
                        </div>
                        <ul style="margin:0; padding-left:18px; font-size:0.88rem; line-height:1.45;">
                            {concepts_html}
                        </ul>
                    </div>
                    <div style="background:#0B0F17; border:1px solid #1E293B; border-radius:6px; padding:12px 16px;">
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#10B981; margin-bottom:8px;">
                            // ARCHITECTURAL MENTAL MODELS
                        </div>
                        <ul style="margin:0 0 10px 0; padding-left:18px; font-size:0.86rem; line-height:1.45;">
                            {models_html}
                        </ul>
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#F87171; margin-bottom:6px;">
                            // ANTI-PATTERNS & TRAPS
                        </div>
                        <ul style="margin:0; padding-left:18px; font-size:0.85rem; line-height:1.45;">
                            {pitfalls_html}
                        </ul>
                    </div>
                </div>
                <div class="capstone-box">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#F59E0B; margin-bottom:4px;">
                        ⚡ PROOF-OF-WORK CAPSTONE DELIVERABLE // {html.escape(capstone.get('title', ''))}
                    </div>
                    <p style="font-size:0.88rem; color:#E2E8F0; margin:0 0 8px 0;">
                        {html.escape(capstone.get('problem_statement', ''))}
                    </p>
                    <ul style="margin:0 0 8px 0; padding-left:18px; font-size:0.84rem;">
                        {cap_reqs_html}
                    </ul>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.76rem; color:#10B981; background:#061E16; padding:6px 10px; border-radius:4px;">
                        <strong>BENCHMARK TEST:</strong> {html.escape(capstone.get('verification_test', ''))}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_toggle, col_res = st.columns([1, 3])
        with col_toggle:
            checked = st.checkbox(
                f"Mark {nid} Complete ({hours}h)",
                value=is_done,
                key=f"chk_{nid}_{roadmap.get('subject', '')[:12]}",
            )
            if checked and nid not in st.session_state["completed_nodes"]:
                st.session_state["completed_nodes"].add(nid)
                st.rerun()
            elif not checked and nid in st.session_state["completed_nodes"]:
                st.session_state["completed_nodes"].discard(nid)
                st.rerun()

        with col_res:
            with st.expander(f"📚 View Curated Resources for {nid} ({len(node.get('resources', []))} items)"):
                for res in node.get("resources", []):
                    cat_key = str(res.get("category", "documentation")).lower()
                    badge_lbl, badge_col = RESOURCE_TYPE_BADGES.get(cat_key, ("REF", "#38BDF8"))
                    st.markdown(
                        f"""
                        <div class="resource-row">
                            <span class="badge-mono" style="background:rgba(56,189,248,0.12); color:{badge_col}; border:1px solid {badge_col};">
                                {badge_lbl}
                            </span>
                            <a href="{html.escape(res.get('url', '#'))}" target="_blank" style="color:#F8FAFC; font-weight:600; text-decoration:underline;">
                                {html.escape(res.get('title', 'Resource'))}
                            </a>
                            <span style="color:#64748B; font-size:0.82rem;"> — {html.escape(res.get('author_or_source', ''))}</span>
                            <div style="color:#94A3B8; font-size:0.84rem; margin-top:4px;">
                                {html.escape(res.get('why_essential', ''))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_resource_matrix_tab(roadmap: Dict[str, Any]) -> None:
    """Renders a consolidated, filterable view of all resources across all modules."""
    all_resources: List[Tuple[str, str, Dict[str, Any]]] = []
    for node in roadmap.get("nodes", []):
        nid = node.get("node_id", "M00")
        ntitle = node.get("title", "")
        for res in node.get("resources", []):
            all_resources.append((nid, ntitle, res))

    categories = sorted({ str(r[2].get("category", "documentation")).lower() for r in all_resources })
    selected_cats = st.multiselect(
        "Filter by Resource Type",
        options=categories,
        default=categories,
    )

    for nid, ntitle, res in all_resources:
        cat = str(res.get("category", "documentation")).lower()
        if selected_cats and cat not in selected_cats:
            continue
        badge_lbl, badge_col = RESOURCE_TYPE_BADGES.get(cat, ("REF", "#38BDF8"))
        st.markdown(
            f"""
            <div class="resource-row">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <span class="badge-mono" style="background:#151C28; color:#F59E0B; border:1px solid #334155;">{html.escape(nid)}</span>
                        <span class="badge-mono" style="background:rgba(56,189,248,0.1); color:{badge_col}; border:1px solid {badge_col};">{badge_lbl}</span>
                        <a href="{html.escape(res.get('url', '#'))}" target="_blank" style="color:#F8FAFC; font-weight:600; font-size:0.96rem;">
                            {html.escape(res.get('title', ''))}
                        </a>
                    </div>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#64748B;">
                        {html.escape(res.get('author_or_source', ''))}
                    </span>
                </div>
                <div style="color:#94A3B8; font-size:0.86rem; margin-top:6px;">
                    {html.escape(res.get('why_essential', ''))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_export_tab(roadmap: Dict[str, Any], completed_ids: set[str]) -> None:
    """Renders one-click exports for Markdown, Graphviz DOT, and raw JSON."""
    md_content = export_roadmap_to_markdown(roadmap, completed_ids)
    dot_content = export_roadmap_to_dot(roadmap)
    json_content = json.dumps(roadmap, indent=2)

    slug = re.sub(r"[^a-z0-9]+", "-", roadmap.get("subject", "roadmap").lower()).strip("-")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇ DOWNLOAD MARKDOWN (.MD)",
            data=md_content,
            file_name=f"{slug}-roadmap.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ DOWNLOAD GRAPHVIZ (.DOT)",
            data=dot_content,
            file_name=f"{slug}-dag.dot",
            mime="text/vnd.graphviz",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "⬇ DOWNLOAD SCHEMA (.JSON)",
            data=json_content,
            file_name=f"{slug}-schema.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("#### Live Markdown Preview")
    st.code(md_content, language="markdown")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="PATHFINDER // AI Roadmap Architect",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_dark_css()
    init_session_state()

    api_key, request_cfg, generate_clicked, load_demo_clicked = render_sidebar()

    if load_demo_clicked:
        st.session_state["active_roadmap"] = get_sample_roadmap()
        st.session_state["completed_nodes"] = {"M01"}
        st.toast("Loaded sample Distributed Systems blueprint.", icon="🧭")

    if generate_clicked:
        is_valid, validation_err = request_cfg.validate()
        if not is_valid:
            st.error(f"Validation Error: {validation_err}")
        elif not api_key.strip():
            st.warning(
                "Please enter your Google Gemini API Key in the left sidebar (or set `GEMINI_API_KEY` in your environment) to synthesize a custom roadmap."
            )
        else:
            with st.status(
                f"Synthesizing curriculum DAG via `{request_cfg.model_id}`...",
                expanded=True,
            ) as status_box:
                st.write("• Validating constraints & time budget...")
                service = GeminiRoadmapService(api_key=api_key)
                try:
                    st.write("• Dispatching structured schema prompt to Gemini 3.8 Flash...")
                    new_roadmap = service.generate_roadmap(request_cfg)
                    st.write("• Verifying topological DAG prerequisites & computing telemetry...")
                    st.session_state["active_roadmap"] = new_roadmap
                    st.session_state["completed_nodes"] = set()
                    st.session_state["roadmap_history"].append(new_roadmap)
                    status_box.update(
                        label="Blueprint synthesis complete!",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    status_box.update(label="Synthesis failed", state="error", expanded=True)
                    st.error(f"Could not generate roadmap: {exc}")

    active_roadmap = st.session_state["active_roadmap"]
    completed_ids: set[str] = st.session_state["completed_nodes"]

    render_telemetry_header(active_roadmap, completed_ids)

    # Interactive Dependency Graph Section
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin: 6px 0 8px 0;">
            <span style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:#F59E0B; letter-spacing:0.06em;">
                // TOPOLOGICAL DEPENDENCY GRAPH (DAG)
            </span>
            <span style="font-family:'JetBrains Mono', monospace; font-size:0.74rem; color:#64748B;">
                GREEN = MASTERED &nbsp;•&nbsp; AMBER = UNLOCKED &nbsp;•&nbsp; SLATE = PREREQ PENDING
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    svg_html = render_svg_topology_graph(active_roadmap, completed_ids)
    st.markdown(svg_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab_modules, tab_resources, tab_export = st.tabs(
        [
            "📐 PHASE ARCHITECTURE & CAPSTONES",
            "📚 CURATED REFERENCE MATRIX",
            "📦 EXPORT & INTEGRATIONS (.MD / .DOT / .JSON)",
        ]
    )

    with tab_modules:
        render_module_cards(active_roadmap, completed_ids)

    with tab_resources:
        render_resource_matrix_tab(active_roadmap)

    with tab_export:
        render_export_tab(active_roadmap, completed_ids)


if __name__ == "__main__":
    main()
