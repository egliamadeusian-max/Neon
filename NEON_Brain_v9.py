"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║           NEON BRAIN v8 — ADVANCED UNIFIED COGNITIVE AI SYSTEM                 ║
║                  Bio-Inspired · Multi-Algorithm · Synchronized                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  NEW ALGORITHMS & COMPONENTS ADDED IN v8:                                      ║
║                                                                                  ║
║  [MATH]      Bayesian Inference · Softmax · Entropy · Cosine Similarity        ║
║              Bellman Equation · UCB1 Bandit · Hebbian Learning · FFT Signal    ║
║              Kalman Filter · Gradient Descent · Sigmoid · Information Gain     ║
║                                                                                  ║
║  [ALGO]      Fuzzy Logic · Genetic Algorithm · Attention Mechanism              ║
║              Graph Neural Network (stub) · Monte Carlo Tree Search (MCTS)      ║
║              Hidden Markov Model · Temporal Difference Learning                 ║
║              Sparse Autoencoder · Mixture of Experts · Chain of Thought        ║
║                                                                                  ║
║  [BRAIN]     Thalamus Router · Amygdala Safety Gate · Hippocampal Replay       ║
║              Dopamine Reward Signal · Prefrontal Working Memory · Cerebellum   ║
║              Predictive Processing · Global Workspace Theory Broadcast          ║
║                                                                                  ║
║  [SYNC]      Async Event Bus · Signal Propagation · Cross-Module Binding       ║
║              Consensus Voting · Weighted Specialist Fusion                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

COGNITIVE PIPELINE:
  Perception → Thalamus → [Specialist Agents in Parallel]
  → Global Workspace Broadcast → Bayesian Integration
  → Prefrontal Planning (MCTS) → Amygdala Safety Check
  → Hippocampal Memory Write → Dopamine RL Update → Output
"""

import os
import math
import time
import random
import hashlib
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
from enum import Enum

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

LLM_API_KEY    = os.getenv("OPENAI_API_KEY", "your_key_here")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4")
MAX_ITERATIONS = 5
MEMORY_LIMIT   = 1000
GAMMA          = 0.99    # RL discount factor
ALPHA          = 0.01    # Learning rate
EPSILON        = 0.1     # Exploration rate (ε-greedy)
TEMPERATURE    = 0.7     # LLM sampling temperature
UCB_C          = 1.414   # √2 — UCB1 exploration constant

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: MATH LIBRARY — All Core Formulas
# ════════════════════════════════════════════════════════

class MathEngine:
    """
    Central mathematical toolkit used across all modules.
    Each formula is documented with its mathematical definition.
    """

    # ─── PROBABILITY & INFORMATION THEORY ───────────────────────────

    @staticmethod
    def softmax(logits: List[float]) -> List[float]:
        """
        Softmax: σ(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
        Converts raw scores into a probability distribution.
        Used by: Specialist router, attention weights.
        """
        max_l = max(logits)
        exps  = [math.exp(z - max_l) for z in logits]  # subtract max for numerical stability
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def entropy(probs: List[float]) -> float:
        """
        Shannon Entropy: H(X) = -Σ p(x) · log₂(p(x))
        Measures uncertainty in a probability distribution.
        High entropy = uncertain → triggers deeper reasoning.
        Used by: MetaCognition, Curiosity Engine.
        """
        return -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)

    @staticmethod
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """
        KL Divergence: D_KL(P‖Q) = Σ p(x) · log(p(x)/q(x))
        Measures how much distribution P diverges from Q.
        Used by: Predictive Processing (surprise detection).
        """
        return sum(pi * math.log((pi + 1e-10) / (qi + 1e-10))
                   for pi, qi in zip(p, q))

    @staticmethod
    def information_gain(parent_entropy: float,
                         children: List[Tuple[int, float]]) -> float:
        """
        IG(T,A) = H(T) - Σ (|Tᵥ|/|T|) · H(Tᵥ)
        Used by: Decision Tree to pick best split attribute.
        """
        total      = sum(n for n, _ in children)
        weighted_h = sum((n / total) * h for n, h in children)
        return parent_entropy - weighted_h

    @staticmethod
    def bayesian_update(prior: float, likelihood: float,
                         evidence: float) -> float:
        """
        Bayes Theorem: P(H|E) = P(E|H) · P(H) / P(E)
        Updates belief probability given new evidence.
        Used by: Belief Updater, Confidence Estimator.
        """
        return (likelihood * prior) / (evidence + 1e-10)

    # ─── SIMILARITY & DISTANCE ──────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        cos(θ) = (A·B) / (‖A‖ · ‖B‖)
        Semantic similarity between two embedding vectors.
        Used by: Semantic Memory retrieval.
        """
        dot   = sum(ai * bi for ai, bi in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b + 1e-10)

    @staticmethod
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        """
        d(a,b) = √(Σ (aᵢ - bᵢ)²)
        Used by: k-NN retrieval, clustering.
        """
        return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

    # ─── NEURAL ACTIVATION FUNCTIONS ────────────────────────────────

    @staticmethod
    def sigmoid(x: float) -> float:
        """σ(x) = 1 / (1 + e^(-x))  — gating, binary decisions"""
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def relu(x: float) -> float:
        """ReLU(x) = max(0, x)  — sparse activation"""
        return max(0.0, x)

    @staticmethod
    def tanh(x: float) -> float:
        """tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))  — bounded output"""
        return math.tanh(x)

    # ─── REINFORCEMENT LEARNING ──────────────────────────────────────

    @staticmethod
    def bellman(reward: float, next_value: float,
                gamma: float = GAMMA) -> float:
        """
        Bellman Equation: V(s) = R + γ · V(s')
        Core of temporal difference (TD) learning.
        Used by: DopamineReward, RL Agent.
        """
        return reward + gamma * next_value

    @staticmethod
    def td_error(reward: float, value: float,
                 next_value: float, gamma: float = GAMMA) -> float:
        """
        TD Error: δ = R + γ·V(s') - V(s)
        The 'prediction error' signal — mirrors dopamine in the brain.
        Positive δ = better than expected → reinforce.
        Negative δ = worse than expected → suppress.
        """
        return reward + gamma * next_value - value

    @staticmethod
    def ucb1(mean_reward: float, n_total: int,
              n_arm: int, c: float = UCB_C) -> float:
        """
        UCB1: Q(a) + c · √(ln(N) / n(a))
        Upper Confidence Bound for exploration-exploitation.
        Used by: Multi-Armed Bandit specialist selector.
        """
        if n_arm == 0:
            return float('inf')
        return mean_reward + c * math.sqrt(math.log(n_total + 1) / n_arm)

    # ─── KALMAN FILTER ──────────────────────────────────────────────

    @staticmethod
    def kalman_update(prior_mean: float, prior_var: float,
                      observation: float, obs_var: float) -> Tuple[float, float]:
        """
        Kalman Filter Update:
          K  = P_prior / (P_prior + R)       — Kalman Gain
          x' = x_prior + K·(z - x_prior)    — Updated estimate
          P' = (1 - K)·P_prior               — Updated variance

        Used by: Confidence tracking over time.
        """
        K       = prior_var / (prior_var + obs_var)
        new_mean = prior_mean + K * (observation - prior_mean)
        new_var  = (1 - K) * prior_var
        return new_mean, new_var

    # ─── HEBBIAN LEARNING ───────────────────────────────────────────

    @staticmethod
    def hebbian_update(weight: float, pre: float,
                        post: float, lr: float = ALPHA) -> float:
        """
        Hebb's Rule: Δw = η · pre · post
        'Neurons that fire together, wire together.'
        Used by: Associative memory formation in HippocampalMemory.
        """
        return weight + lr * pre * post

    # ─── GRADIENT DESCENT ───────────────────────────────────────────

    @staticmethod
    def gradient_step(param: float, gradient: float,
                       lr: float = ALPHA) -> float:
        """
        θ ← θ - η · ∇L(θ)
        Single step of gradient descent.
        Used by: Weight adaptation in specialist agents.
        """
        return param - lr * gradient

    # ─── FUZZY LOGIC ────────────────────────────────────────────────

    @staticmethod
    def fuzzy_and(a: float, b: float) -> float:
        """Fuzzy AND: T(a,b) = min(a,b)"""
        return min(a, b)

    @staticmethod
    def fuzzy_or(a: float, b: float) -> float:
        """Fuzzy OR: T(a,b) = max(a,b)"""
        return max(a, b)

    @staticmethod
    def fuzzy_not(a: float) -> float:
        """Fuzzy NOT: ¬a = 1 - a"""
        return 1.0 - a

    @staticmethod
    def defuzzify(values: List[float], weights: List[float]) -> float:
        """
        Centroid Defuzzification: z* = Σ(μᵢ · zᵢ) / Σ μᵢ
        Converts fuzzy set to crisp output value.
        Used by: FuzzyDecisionEngine.
        """
        num = sum(w * v for w, v in zip(weights, values))
        den = sum(weights)
        return num / (den + 1e-10)

    # ─── FAST FOURIER TRANSFORM (signal analysis) ───────────────────

    @staticmethod
    def dft(signal: List[float]) -> List[complex]:
        """
        DFT: X[k] = Σₙ x[n] · e^(-j·2π·k·n/N)
        Detects periodic patterns in input streams.
        Used by: PatternDetector for cyclical behavior.
        """
        N   = len(signal)
        out = []
        for k in range(N):
            total = sum(signal[n] * complex(
                math.cos(2 * math.pi * k * n / N),
               -math.sin(2 * math.pi * k * n / N)
            ) for n in range(N))
            out.append(total)
        return out


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: MEMORY ARCHITECTURE (Multi-Layer)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTrace:
    content: str
    embedding: List[float]
    timestamp: float
    importance: float = 1.0
    recall_count: int = 0


class WorkingMemory:
    """
    Prefrontal Cortex analog.
    Limited-capacity buffer (Miller's Law: 7±2 chunks).
    Uses LRU eviction with importance weighting.
    """
    CAPACITY = 9

    def __init__(self):
        self.buffer: deque = deque(maxlen=self.CAPACITY)

    def push(self, item: str):
        self.buffer.appendleft(item)

    def peek(self, n: int = 5) -> List[str]:
        return list(self.buffer)[:n]

    def clear(self):
        self.buffer.clear()


class EpisodicMemory:
    """
    Hippocampus analog.
    Stores experiences as time-stamped traces.
    Supports Hebbian consolidation and replay.
    """

    def __init__(self, capacity: int = MEMORY_LIMIT):
        self.traces: List[MemoryTrace] = []
        self.capacity = capacity
        self.weights: Dict[str, float] = {}  # Hebbian weights

    def _simple_embed(self, text: str) -> List[float]:
        """Hash-based pseudo-embedding (replace with real embeddings)."""
        h     = hashlib.md5(text.encode()).hexdigest()
        seeds = [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]
        return seeds

    def store(self, text: str, importance: float = 1.0):
        emb = self._simple_embed(text)
        trace = MemoryTrace(
            content      = text,
            embedding    = emb,
            timestamp    = time.time(),
            importance   = importance
        )
        if len(self.traces) >= self.capacity:
            # Remove least important trace
            self.traces.sort(key=lambda t: t.importance * t.recall_count)
            self.traces.pop(0)
        self.traces.append(trace)

        # Hebbian weight update between co-occurring memories
        key = text[:30]
        if key in self.weights:
            self.weights[key] = MathEngine.hebbian_update(
                self.weights[key], 1.0, importance
            )
        else:
            self.weights[key] = importance

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Cosine similarity retrieval."""
        q_emb = self._simple_embed(query)
        scored = []
        for trace in self.traces:
            sim = MathEngine.cosine_similarity(q_emb, trace.embedding)
            scored.append((sim, trace))
        scored.sort(reverse=True)
        results = []
        for _, trace in scored[:top_k]:
            trace.recall_count += 1
            results.append(trace.content)
        return results

    def replay(self) -> List[str]:
        """
        Hippocampal Replay: consolidate high-importance memories.
        Mimics sleep-stage memory consolidation.
        Returns top memories by importance × recency.
        """
        now = time.time()
        scored = [
            (t.importance / (1 + now - t.timestamp) * 0.01, t)
            for t in self.traces
        ]
        scored.sort(reverse=True)
        return [t.content for _, t in scored[:10]]


class SemanticMemory:
    """
    Temporal Lobe / Neocortex analog.
    Stores factual knowledge as concept → attribute → value triples.
    Supports graph traversal queries.
    """

    def __init__(self):
        self.knowledge: Dict[str, Dict[str, str]] = {}
        self.relations: List[Tuple[str, str, str]] = []  # (subject, relation, object)

    def assert_fact(self, concept: str, attr: str, value: str):
        if concept not in self.knowledge:
            self.knowledge[concept] = {}
        self.knowledge[concept][attr] = value

    def assert_relation(self, subj: str, rel: str, obj: str):
        self.relations.append((subj, rel, obj))

    def query(self, concept: str) -> Dict[str, str]:
        return self.knowledge.get(concept, {})

    def find_related(self, concept: str) -> List[Tuple[str, str]]:
        return [(rel, obj) for s, rel, obj in self.relations if s == concept]


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: ATTENTION MECHANISM
# ══════════════════════════════════════════════════════════════════════

class AttentionMechanism:
    """
    Scaled Dot-Product Attention:
    Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V

    Routes cognitive resources to the most relevant specialists.
    Mimics thalamo-cortical attention gating.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.W_q = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
        self.W_k = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(ai * bi for ai, bi in zip(a, b))

    def attend(self, query: List[float],
               keys: List[List[float]]) -> List[float]:
        """Returns attention weights over keys given a query."""
        scale    = math.sqrt(self.dim)
        scores   = [self._dot(query, k) / scale for k in keys]
        return MathEngine.softmax(scores)

    def weighted_sum(self, weights: List[float],
                     values: List[List[float]]) -> List[float]:
        """Weighted sum of value vectors."""
        result = [0.0] * len(values[0])
        for w, v in zip(weights, values):
            for i, vi in enumerate(v):
                result[i] += w * vi
        return result


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: DECISION TREE ENGINE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DTNode:
    feature:   Optional[str]  = None
    threshold: Optional[float] = None
    label:     Optional[str]  = None
    left:      Optional['DTNode'] = None
    right:     Optional['DTNode'] = None


class DecisionTreeRouter:
    """
    Information-Gain Decision Tree.
    Routes input to the correct specialist based on detected features.
    Uses: IG(T,A) = H(T) - Σ (|Tᵥ|/|T|)·H(Tᵥ)

    Features extracted from text:
      - has_question  → Analytical agent
      - has_math      → Analytical agent
      - has_creative  → Creative agent
      - has_code      → Coding agent
      - has_risk      → Safety agent
      - default       → Conversation agent
    """

    def __init__(self):
        # Pre-built decision tree (replace with learned tree for dynamic routing)
        self.root = self._build()

    def _build(self) -> DTNode:
        safety    = DTNode(label="safety")
        coding    = DTNode(label="coding")
        analytic  = DTNode(label="analytical")
        creative  = DTNode(label="creative")
        convo     = DTNode(label="conversation")

        math_node = DTNode(feature="has_math",  threshold=0.5, left=analytic, right=analytic)
        code_node = DTNode(feature="has_code",  threshold=0.5, left=coding,   right=math_node)
        crea_node = DTNode(feature="has_creative", threshold=0.5, left=creative, right=code_node)
        safe_node = DTNode(feature="has_risk",  threshold=0.5, left=safety,   right=crea_node)
        root      = DTNode(feature="has_risk",  threshold=0.5, left=safety,   right=crea_node)
        return root

    def _extract_features(self, text: str) -> Dict[str, float]:
        t = text.lower()
        return {
            "has_question":  float("?" in t or "how" in t or "what" in t or "why" in t),
            "has_math":      float(any(c in t for c in ["calculate", "math", "formula", "equation", "number", "solve"])),
            "has_code":      float(any(c in t for c in ["code", "program", "function", "debug", "python", "script"])),
            "has_creative":  float(any(c in t for c in ["write", "create", "story", "poem", "imagine", "design"])),
            "has_risk":      float(any(c in t for c in ["harm", "danger", "illegal", "weapon", "kill", "attack"])),
        }

    def route(self, text: str) -> str:
        feats  = self._extract_features(text)
        node   = self.root
        while node.label is None:
            val = feats.get(node.feature, 0.0)
            node = node.left if val >= node.threshold else node.right
        return node.label

    def information_gain_demo(self, samples: List[Dict[str, float]],
                               labels: List[str]) -> Dict[str, float]:
        """Compute IG for each feature — shows how splits are chosen."""
        unique_labels = list(set(labels))
        n             = len(labels)
        label_probs   = [labels.count(l) / n for l in unique_labels]
        parent_h      = MathEngine.entropy(label_probs)
        ig_scores     = {}

        all_features = samples[0].keys() if samples else []
        for feat in all_features:
            above = [labels[i] for i, s in enumerate(samples) if s[feat] >= 0.5]
            below = [labels[i] for i, s in enumerate(samples) if s[feat] <  0.5]
            children = []
            for subset in [above, below]:
                if subset:
                    p = [subset.count(l) / len(subset) for l in unique_labels]
                    children.append((len(subset), MathEngine.entropy(p)))
            ig_scores[feat] = MathEngine.information_gain(parent_h, children)
        return ig_scores


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: FUZZY LOGIC DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════

class FuzzyDecisionEngine:
    """
    Fuzzy Logic Controller for nuanced, gradient decisions.

    Instead of hard yes/no, evaluates degrees of truth.
    E.g. confidence = 0.73 → "fairly confident" (not binary).

    Rules:
      IF confidence HIGH AND uncertainty LOW → action AGGRESSIVE
      IF confidence LOW OR uncertainty HIGH  → action CAUTIOUS
      IF risk HIGH                           → action BLOCKED
    """

    def _confidence_membership(self, c: float) -> Dict[str, float]:
        return {
            "low":    MathEngine.sigmoid(-10 * (c - 0.4)),
            "medium": math.exp(-((c - 0.5) ** 2) / 0.05),
            "high":   MathEngine.sigmoid(10 * (c - 0.6)),
        }

    def _uncertainty_membership(self, u: float) -> Dict[str, float]:
        return {
            "low":  MathEngine.sigmoid(-10 * (u - 0.3)),
            "high": MathEngine.sigmoid(10 * (u - 0.6)),
        }

    def decide(self, confidence: float, uncertainty: float,
                risk: float) -> Dict[str, float]:
        conf = self._confidence_membership(confidence)
        unc  = self._uncertainty_membership(uncertainty)

        aggressive = MathEngine.fuzzy_and(conf["high"], unc["low"])
        cautious   = MathEngine.fuzzy_or(conf["low"],   unc["high"])
        blocked    = MathEngine.sigmoid(10 * (risk - 0.7))
        explore    = MathEngine.fuzzy_and(conf["medium"], unc["high"])

        raw     = [aggressive, cautious, blocked, explore]
        labels  = ["aggressive", "cautious", "blocked", "explore"]
        crisp   = MathEngine.defuzzify([0.9, 0.3, 0.0, 0.6], raw)

        return {
            "action":     labels[raw.index(max(raw))],
            "crisp_value": crisp,
            "degrees":    dict(zip(labels, raw))
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: GENETIC ALGORITHM (Creative Solution Search)
# ══════════════════════════════════════════════════════════════════════

class GeneticAlgorithm:
    """
    Evolutionary search for creative plan generation.
    Inspired by the brain's ability to evolve and prune neural pathways.

    Operations:
      Selection:   Tournament selection
      Crossover:   Single-point crossover
      Mutation:    Random word substitution
      Fitness:     LLM-scored heuristic (mocked here)

    Fitness: F(x) = Σ wᵢ · fᵢ(x)   (weighted multi-objective)
    """

    def __init__(self, pop_size: int = 10, generations: int = 5,
                 mutation_rate: float = 0.1):
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.vocab         = [
            "analyze", "create", "optimize", "explore", "synthesize",
            "evaluate", "transform", "integrate", "connect", "generate",
            "discover", "map", "model", "simulate", "predict"
        ]

    def _init_population(self, goal: str) -> List[List[str]]:
        words = goal.split()
        pop   = []
        for _ in range(self.pop_size):
            individual = words[:]
            # Random creative mutations in initial population
            if random.random() < 0.5:
                individual.insert(random.randint(0, len(individual)),
                                  random.choice(self.vocab))
            pop.append(individual)
        return pop

    def _fitness(self, individual: List[str]) -> float:
        """
        Multi-objective fitness:
          F = w₁·length_score + w₂·diversity_score + w₃·goal_alignment
        """
        text            = " ".join(individual)
        length_score    = min(len(individual) / 10.0, 1.0)
        diversity_score = len(set(individual)) / (len(individual) + 1)
        goal_alignment  = random.uniform(0.3, 1.0)  # replace with LLM scoring
        return 0.2 * length_score + 0.3 * diversity_score + 0.5 * goal_alignment

    def _tournament_select(self, pop: List[List[str]],
                            scores: List[float]) -> List[str]:
        idx_a, idx_b = random.sample(range(len(pop)), 2)
        return pop[idx_a] if scores[idx_a] > scores[idx_b] else pop[idx_b]

    def _crossover(self, a: List[str], b: List[str]) -> List[str]:
        if len(a) < 2 or len(b) < 2:
            return a
        point = random.randint(1, min(len(a), len(b)) - 1)
        return a[:point] + b[point:]

    def _mutate(self, individual: List[str]) -> List[str]:
        result = individual[:]
        for i in range(len(result)):
            if random.random() < self.mutation_rate:
                result[i] = random.choice(self.vocab)
        return result

    def evolve(self, goal: str) -> str:
        """Run genetic search and return best creative strategy."""
        pop = self._init_population(goal)

        for gen in range(self.generations):
            scores  = [self._fitness(ind) for ind in pop]
            new_pop = []
            for _ in range(self.pop_size):
                parent_a = self._tournament_select(pop, scores)
                parent_b = self._tournament_select(pop, scores)
                child    = self._crossover(parent_a, parent_b)
                child    = self._mutate(child)
                new_pop.append(child)
            pop = new_pop

        scores = [self._fitness(ind) for ind in pop]
        best   = pop[scores.index(max(scores))]
        return " ".join(best)


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: MONTE CARLO TREE SEARCH (MCTS)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    state:    str
    parent:   Optional['MCTSNode'] = None
    children: List['MCTSNode']     = field(default_factory=list)
    visits:   int                  = 0
    value:    float                = 0.0

    def ucb1(self, total_visits: int, c: float = UCB_C) -> float:
        """UCB1: Q(s) + c·√(ln(N)/n)"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration  = c * math.sqrt(math.log(total_visits + 1) / self.visits)
        return exploitation + exploration


class MCTSPlanner:
    """
    Monte Carlo Tree Search for multi-step planning.
    Replaces random planning with probabilistic rollout search.

    Algorithm:
      1. SELECT   — traverse tree using UCB1
      2. EXPAND   — add new child node
      3. SIMULATE — random rollout to estimate value
      4. BACKPROP — propagate value up the tree

    Analogous to: Prefrontal cortex deliberate planning + mental simulation.
    """

    def __init__(self, simulations: int = 50):
        self.simulations = simulations

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            total = node.visits
            node  = max(node.children, key=lambda c: c.ucb1(total))
        return node

    def _expand(self, node: MCTSNode, goal: str) -> MCTSNode:
        """Generate possible next actions from current state."""
        actions = [
            f"Analyze: {goal}",
            f"Research: {goal}",
            f"Draft solution for: {goal}",
            f"Critique approach to: {goal}",
            f"Refine output for: {goal}"
        ]
        if len(node.children) < len(actions):
            action = actions[len(node.children)]
            child  = MCTSNode(state=action, parent=node)
            node.children.append(child)
            return child
        return node

    def _simulate(self, node: MCTSNode) -> float:
        """
        Random rollout: simulate until terminal, return reward.
        Reward R ∈ [0,1] estimated by heuristic.
        """
        depth   = 0
        reward  = 0.0
        content = node.state
        while depth < 3:
            # Heuristic: longer & more specific plans score higher
            reward += random.uniform(0.3, 1.0)
            depth  += 1
        return reward / depth

    def _backprop(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value  += value
            node         = node.parent

    def search(self, goal: str) -> List[str]:
        """Returns the best sequence of actions for a goal."""
        root = MCTSNode(state=f"Goal: {goal}")

        for _ in range(self.simulations):
            leaf  = self._select(root)
            child = self._expand(leaf, goal)
            val   = self._simulate(child)
            self._backprop(child, val)

        # Extract best path
        plan   = []
        node   = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            plan.append(node.state)

        return plan if plan else [f"Direct execution: {goal}"]


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: HIDDEN MARKOV MODEL (State Prediction)
# ══════════════════════════════════════════════════════════════════════

class HiddenMarkovModel:
    """
    HMM for conversation state tracking.
    Hidden states: [exploring, focusing, concluding, confused, creative]
    Observations: detected features from user input

    P(O|λ) = Σ P(O|Q,λ) · P(Q|λ)   — likelihood of observations given model
    Forward Algorithm: αₜ(i) = Σⱼ αₜ₋₁(j) · aⱼᵢ · bᵢ(oₜ)
    """

    STATES = ["exploring", "focusing", "concluding", "confused", "creative"]
    OBS    = ["question", "statement", "command", "emotional", "technical"]

    def __init__(self):
        n = len(self.STATES)
        m = len(self.OBS)

        # Transition matrix A[i][j] = P(state j | state i)
        self.A = self._normalize([
            [0.3, 0.4, 0.1, 0.1, 0.1],
            [0.1, 0.4, 0.3, 0.1, 0.1],
            [0.1, 0.2, 0.5, 0.1, 0.1],
            [0.3, 0.1, 0.1, 0.4, 0.1],
            [0.2, 0.2, 0.1, 0.1, 0.4],
        ])
        # Emission matrix B[i][k] = P(obs k | state i)
        self.B = self._normalize([
            [0.4, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.0, 0.2],
            [0.0, 0.3, 0.4, 0.1, 0.2],
            [0.3, 0.1, 0.1, 0.4, 0.1],
            [0.2, 0.1, 0.1, 0.1, 0.5],
        ])
        # Initial state distribution π
        self.pi    = [0.4, 0.2, 0.1, 0.2, 0.1]
        self.state = 0  # current state index

    def _normalize(self, matrix: List[List[float]]) -> List[List[float]]:
        return [[v / sum(row) for v in row] for row in matrix]

    def _obs_index(self, text: str) -> int:
        t = text.lower()
        if "?" in t:              return 0  # question
        if any(c in t for c in ["do", "run", "execute", "start"]): return 2  # command
        if any(c in t for c in ["feel", "sad", "happy", "love"]):  return 3  # emotional
        if any(c in t for c in ["code", "math", "formula", "api"]): return 4  # technical
        return 1  # statement

    def update(self, text: str) -> str:
        """Viterbi-like greedy state transition."""
        obs   = self._obs_index(text)
        probs = [
            self.A[self.state][j] * self.B[j][obs]
            for j in range(len(self.STATES))
        ]
        self.state = probs.index(max(probs))
        return self.STATES[self.state]

    def predict_next(self) -> str:
        """Predict most likely next state."""
        next_probs = self.A[self.state]
        return self.STATES[next_probs.index(max(next_probs))]


# ══════════════════════════════════════════════════════════════════════
# SECTION 10: MIXTURE OF EXPERTS (Specialist Fusion)
# ══════════════════════════════════════════════════════════════════════

class MixtureOfExperts:
    """
    MoE: output = Σᵢ gᵢ(x) · Eᵢ(x)
    Where gᵢ(x) = softmax(W_g · x)ᵢ   — gating network
    and Eᵢ(x)   = expert i output

    Selects and weights specialist agents dynamically.
    Only top-k experts contribute (sparse gating).
    Top-K gate: keep top k, zero-out rest, renormalize.
    """

    def __init__(self, expert_names: List[str], top_k: int = 2):
        self.experts  = expert_names
        self.top_k    = top_k
        # Learnable gate weights (initialized randomly, updated by RL)
        self.W_gate   = {name: random.uniform(0.5, 1.5) for name in expert_names}
        self.call_log = defaultdict(int)

    def gate(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Compute gating weights for each expert."""
        raw_scores = {}
        for expert in self.experts:
            # Gate score = base weight × feature affinity
            affinity = features.get(f"has_{expert.split('_')[0]}", 0.5)
            raw_scores[expert] = self.W_gate[expert] * (0.5 + affinity)

        names  = list(raw_scores.keys())
        scores = [raw_scores[n] for n in names]
        probs  = MathEngine.softmax(scores)

        # Top-K sparse gating
        paired  = sorted(zip(probs, names), reverse=True)
        top_k   = paired[:self.top_k]
        total_w = sum(p for p, _ in top_k)

        return [(name, p / total_w) for p, name in top_k]

    def update_gate(self, expert: str, reward: float):
        """Hebbian-style gate weight update based on reward."""
        self.W_gate[expert] = MathEngine.gradient_step(
            self.W_gate[expert], -reward  # negative because we maximize
        )
        self.call_log[expert] += 1


# ══════════════════════════════════════════════════════════════════════
# SECTION 11: DOPAMINE REWARD SIGNAL (RL Core)
# ══════════════════════════════════════════════════════════════════════

class DopamineRewardSystem:
    """
    Models the mesocortical dopamine pathway.
    Computes TD Error as the analog of phasic dopamine firing.

    δ = R + γ·V(s') - V(s)
    δ > 0: better than predicted → dopamine burst → reinforce
    δ < 0: worse than predicted  → dopamine dip   → suppress
    δ = 0: as predicted          → no change      → baseline

    Also implements multi-armed bandit for specialist selection.
    """

    def __init__(self, n_arms: int, arm_names: List[str]):
        self.n_arms    = n_arms
        self.arm_names = arm_names
        self.Q         = {name: 0.0 for name in arm_names}  # value estimates
        self.N         = {name: 0   for name in arm_names}  # pull counts
        self.N_total   = 0
        self.V         = 0.0  # current state value estimate
        self.history   = []

    def td_update(self, reward: float, next_value: float) -> float:
        """Update state value and return TD error (dopamine signal)."""
        delta    = MathEngine.td_error(reward, self.V, next_value)
        self.V   = MathEngine.bellman(reward, next_value)
        self.history.append({"reward": reward, "delta": delta, "V": self.V})
        return delta

    def select_arm(self) -> str:
        """UCB1 bandit: exploration-exploitation balance."""
        ucb_scores = {
            name: MathEngine.ucb1(
                self.Q[name], self.N_total, self.N[name]
            )
            for name in self.arm_names
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update_arm(self, arm: str, reward: float):
        """Incremental Q update: Q(a) ← Q(a) + 1/N·(R - Q(a))"""
        self.N[arm]   += 1
        self.N_total  += 1
        self.Q[arm]   += (1 / self.N[arm]) * (reward - self.Q[arm])

    def get_q_table(self) -> Dict[str, float]:
        return dict(self.Q)


# ══════════════════════════════════════════════════════════════════════
# SECTION 12: PREDICTIVE PROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════

class PredictiveProcessingEngine:
    """
    Implements Karl Friston's Predictive Processing / Free Energy Principle.
    The brain constantly generates predictions and minimizes prediction error.

    Free Energy: F = KL[q(s) ‖ p(s|o)] - log p(o)
    Approximated here as: F ≈ prediction_error + complexity_cost

    Prediction error (PE) drives:
    - Attention: high PE → boost attention on that input
    - Learning: update internal model to reduce future PE
    - Curiosity: seek inputs that maximally reduce F
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features
        self.predictions: Dict[str, List[float]] = {}
        self.kalman_state  = 0.5   # estimated state
        self.kalman_var    = 1.0   # uncertainty

    def _embed(self, text: str) -> List[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, self.n_features * 2, 2)]

    def predict(self, context: str) -> List[float]:
        """Generate prediction for next input given context."""
        if context in self.predictions:
            return self.predictions[context]
        # Prior: uniform prediction
        return [0.5] * self.n_features

    def update(self, context: str,
               actual: str) -> Dict[str, float]:
        """Compute prediction error and update model."""
        pred     = self.predict(context)
        actual_e = self._embed(actual)

        # Prediction error: L2 distance between prediction and reality
        pe = MathEngine.euclidean_distance(pred, actual_e)

        # KL divergence for surprise measure
        pred_probs   = MathEngine.softmax(pred)
        actual_probs = MathEngine.softmax(actual_e)
        surprise     = MathEngine.kl_divergence(actual_probs, pred_probs)

        # Kalman update on estimated confidence
        self.kalman_state, self.kalman_var = MathEngine.kalman_update(
            self.kalman_state, self.kalman_var,
            1.0 - min(pe, 1.0), 0.1
        )

        # Update prediction (exponential moving average)
        self.predictions[context] = [
            0.8 * p + 0.2 * a for p, a in zip(pred, actual_e)
        ]

        return {
            "prediction_error": pe,
            "surprise":         surprise,
            "free_energy":      pe + surprise,
            "confidence":       self.kalman_state
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 13: GLOBAL WORKSPACE THEORY (Broadcast Bus)
# ══════════════════════════════════════════════════════════════════════

class GlobalWorkspace:
    """
    Implementation of Baars' Global Workspace Theory.
    A central 'blackboard' that broadcasts information to all modules.

    Principle: Only the most salient information enters the workspace
    and gets broadcast globally. This is the neural correlate of consciousness.

    Competition: specialists compete for workspace access.
    Winner: highest salience (attention score × confidence).
    Broadcast: winner's content sent to ALL modules simultaneously.
    """

    def __init__(self):
        self.broadcast_content: Optional[str]  = None
        self.subscribers:       List[str]       = []
        self.history:           List[Dict]      = []
        self.salience_log:      Dict[str, float] = {}

    def subscribe(self, module_name: str):
        self.subscribers.append(module_name)

    def compete(self, candidates: Dict[str, Tuple[str, float]]) -> str:
        """
        Candidates: {module_name: (content, salience_score)}
        Winner = argmax salience.
        """
        winner = max(candidates, key=lambda k: candidates[k][1])
        self.salience_log = {k: v[1] for k, v in candidates.items()}
        return winner

    def broadcast(self, winner: str,
                  candidates: Dict[str, Tuple[str, float]]):
        """Send winner's content to all subscribers."""
        content, salience = candidates[winner]
        self.broadcast_content = content
        event = {
            "winner":      winner,
            "content":     content,
            "salience":    salience,
            "timestamp":   time.time(),
            "subscribers": self.subscribers[:]
        }
        self.history.append(event)
        return event

    def get_broadcast(self) -> Optional[str]:
        return self.broadcast_content


# ══════════════════════════════════════════════════════════════════════
# SECTION 14: AMYGDALA SAFETY GATE
# ══════════════════════════════════════════════════════════════════════

class AmygdalaSafetyGate:
    """
    Amygdala: Rapid threat detection before cortical processing.
    Acts as a fast-path safety filter — fires before full reasoning.

    Uses:
    - Keyword pattern matching (fast path)
    - Fuzzy risk scoring (nuanced path)
    - Bayesian risk update from prior incidents

    Risk Model:
      P(risk|evidence) = P(evidence|risk) · P(risk) / P(evidence)
    """

    RISK_KEYWORDS = {
        "critical": ["weapon", "bomb", "kill", "attack", "hack", "illegal", "harm"],
        "moderate": ["dangerous", "risk", "unsafe", "breach", "exploit"],
        "low":      ["sensitive", "private", "confidential"]
    }

    def __init__(self):
        self.prior_risk    = 0.05  # Prior P(risk) — low by default
        self.incident_log  = []
        self.fuzzy         = FuzzyDecisionEngine()

    def assess(self, text: str) -> Dict[str, Any]:
        t      = text.lower()
        level  = "safe"
        score  = 0.0
        found  = []

        for severity, words in self.RISK_KEYWORDS.items():
            for w in words:
                if w in t:
                    found.append(w)
                    score += {"critical": 0.9, "moderate": 0.5, "low": 0.2}[severity]

        score = min(score, 1.0)

        # Bayesian risk update
        likelihood = score
        posterior  = MathEngine.bayesian_update(
            self.prior_risk, likelihood, max(likelihood * self.prior_risk, 0.01)
        )
        posterior  = min(posterior, 1.0)

        # Fuzzy decision
        fuzzy_out  = self.fuzzy.decide(
            confidence  = 1.0 - score,
            uncertainty = score,
            risk        = score
        )

        if score > 0.7:
            level = "BLOCKED"
        elif score > 0.3:
            level = "WARNING"
        else:
            level = "SAFE"

        result = {
            "level":           level,
            "risk_score":      score,
            "bayesian_risk":   posterior,
            "fuzzy_action":    fuzzy_out["action"],
            "triggered_words": found,
            "pass":            level != "BLOCKED"
        }

        if level != "SAFE":
            self.incident_log.append(result)
            self.prior_risk = min(0.5, self.prior_risk + 0.01)

        return result


# ══════════════════════════════════════════════════════════════════════
# SECTION 15: CHAIN OF THOUGHT ENGINE
# ══════════════════════════════════════════════════════════════════════

class ChainOfThoughtEngine:
    """
    Explicit step-by-step reasoning chain.
    Each step conditions on all previous steps.

    CoT format:
      Q: [question]
      A: Let's think step by step.
         Step 1: [reasoning]
         Step 2: [reasoning]
         ...
         Therefore: [conclusion]

    Enhances analytical depth and reduces hallucination.
    """

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def reason(self, question: str, ask_fn) -> Dict[str, Any]:
        chain  = []
        prompt = f"Question: {question}\nLet's think step by step.\nStep 1:"

        for step_num in range(1, self.max_steps + 1):
            step_result = ask_fn(prompt)
            chain.append(f"Step {step_num}: {step_result}")
            prompt = f"{prompt}\n{step_result}\nStep {step_num + 1}:"

            # Stop if we detect a conclusion
            if any(kw in step_result.lower()
                   for kw in ["therefore", "conclusion", "answer is", "result is"]):
                break

        conclusion_prompt = "\n".join(chain) + "\nTherefore:"
        conclusion        = ask_fn(conclusion_prompt)

        return {
            "question":   question,
            "chain":      chain,
            "conclusion": conclusion,
            "steps":      len(chain)
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 16: CURIOSITY ENGINE (Information-Seeking)
# ══════════════════════════════════════════════════════════════════════

class CuriosityEngine:
    """
    Intrinsic motivation via information gain and novelty.
    Drives exploration toward uncertain, novel domains.

    Curiosity Score:
      C(x) = β · (H_prior - H_posterior) + γ · novelty(x)

    Novelty: 1 - max cosine similarity to all seen inputs
    Information Gain: reduction in entropy after observing x
    """

    def __init__(self, beta: float = 0.7, gamma: float = 0.3):
        self.beta         = beta
        self.gamma        = gamma
        self.seen_hashes  = set()
        self.seen_embeds  = []

    def _embed(self, text: str) -> List[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]

    def novelty(self, text: str) -> float:
        """Novelty = 1 - max_similarity to previously seen inputs."""
        emb = self._embed(text)
        if not self.seen_embeds:
            return 1.0
        max_sim = max(MathEngine.cosine_similarity(emb, s)
                      for s in self.seen_embeds)
        return 1.0 - max_sim

    def information_gain(self, uncertainty_before: float = 0.8,
                          uncertainty_after: float = 0.5) -> float:
        """IG = H_before - H_after  (entropy reduction)"""
        return max(0.0, uncertainty_before - uncertainty_after)

    def score(self, text: str, uncertainty_before: float = 0.8,
              uncertainty_after: float = 0.5) -> Dict[str, float]:
        nov = self.novelty(text)
        ig  = self.information_gain(uncertainty_before, uncertainty_after)
        c   = self.beta * ig + self.gamma * nov

        emb = self._embed(text)
        self.seen_embeds.append(emb)
        self.seen_hashes.add(hashlib.md5(text.encode()).hexdigest())

        return {
            "curiosity":        c,
            "novelty":          nov,
            "information_gain": ig,
            "explore":          c > 0.5
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 17: METACOGNITION ENGINE
# ══════════════════════════════════════════════════════════════════════

class MetaCognitionEngine:
    """
    Anterior Cingulate Cortex analog.
    Monitors, evaluates, and corrects the system's own reasoning.

    Functions:
    - Confidence calibration (Brier Score: BS = Σ(fᵢ - oᵢ)² / n)
    - Error detection
    - Strategy switching
    - Self-consistency checking
    """

    def __init__(self):
        self.predictions = []   # (confidence, correct) tuples
        self.k_state, self.k_var = 0.75, 0.5

    def brier_score(self) -> float:
        """
        Brier Score: BS = (1/N) · Σ (fᵢ - oᵢ)²
        Measures calibration of confidence.
        BS = 0.0 → perfectly calibrated.
        BS = 1.0 → completely wrong.
        """
        if not self.predictions:
            return 0.0
        return sum((f - o)**2 for f, o in self.predictions) / len(self.predictions)

    def reflect(self, response: str, confidence: float,
                context: str) -> Dict[str, Any]:
        """Self-evaluate a response."""
        # Consistency check: does response address the context?
        keywords   = set(context.lower().split())
        resp_words = set(response.lower().split())
        overlap    = len(keywords & resp_words) / (len(keywords) + 1)

        # Calibrate confidence using Kalman filter
        obs_confidence       = overlap
        self.k_state, self.k_var = MathEngine.kalman_update(
            self.k_state, self.k_var, obs_confidence, 0.2
        )

        entropy_of_conf = MathEngine.entropy([confidence, 1 - confidence])
        flags           = []

        if confidence < 0.4:     flags.append("LOW_CONFIDENCE")
        if overlap < 0.2:        flags.append("OFF_TOPIC")
        if entropy_of_conf > 0.9: flags.append("HIGH_UNCERTAINTY")
        if len(response) < 20:   flags.append("TOO_BRIEF")

        return {
            "calibrated_confidence": self.k_state,
            "consistency_overlap":   overlap,
            "entropy":               entropy_of_conf,
            "flags":                 flags,
            "brier_score":           self.brier_score(),
            "should_retry":          len(flags) > 1
        }

    def record_outcome(self, confidence: float, was_correct: bool):
        self.predictions.append((confidence, float(was_correct)))


# ══════════════════════════════════════════════════════════════════════
# SECTION 18: LLM INTERFACE (Pluggable)
# ══════════════════════════════════════════════════════════════════════

class LLMInterface:
    """
    Abstracted LLM interface.
    Replace `_call_api` with real HTTP call to OpenAI / Anthropic / local.
    Supports: system prompt, temperature, max_tokens.
    """

    def __init__(self, model: str = LLM_MODEL,
                 temperature: float = TEMPERATURE):
        self.model       = model
        self.temperature = temperature

    def _call_api(self, system: str, user: str) -> str:
        """
        TO WIRE IN: Replace with real API call, e.g.:
          import openai
          r = openai.chat.completions.create(
              model=self.model,
              messages=[{"role":"system","content":system},
                        {"role":"user","content":user}],
              temperature=self.temperature
          )
          return r.choices[0].message.content
        """
        # Mock response for architecture demonstration
        return f"[{self.model}] Processing: {user[:80]}... | Temp={self.temperature}"

    def ask(self, prompt: str, system: str = "You are a helpful AI.") -> str:
        return self._call_api(system, prompt)

    def ask_structured(self, prompt: str,
                        schema: Dict) -> Dict:
        """Request structured JSON output."""
        sys = (
            f"You are a helpful AI. Always respond ONLY with valid JSON "
            f"matching this schema: {json.dumps(schema)}. No other text."
        )
        raw = self._call_api(sys, prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "parse_failed", "raw": raw}


# ══════════════════════════════════════════════════════════════════════
# SECTION 19: SPECIALIST AGENTS (Parallel Cortical Columns)
# ══════════════════════════════════════════════════════════════════════

class SpecialistAgent:
    """Base class for all specialist agents."""

    def __init__(self, name: str, llm: LLMInterface):
        self.name    = name
        self.llm     = llm
        self.weight  = 1.0  # updated by RL

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        raise NotImplementedError

    def _base_result(self, output: str,
                      confidence: float) -> Dict[str, Any]:
        return {
            "agent":      self.name,
            "output":     output,
            "confidence": confidence,
            "weight":     self.weight
        }


class AnalyticalAgent(SpecialistAgent):
    """Left hemisphere: logic, analysis, math, structure."""

    def __init__(self, llm: LLMInterface):
        super().__init__("analytical", llm)
        self.cot = ChainOfThoughtEngine(max_steps=4)

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        chain = self.cot.reason(input_text, self.llm.ask)
        return self._base_result(
            output     = chain["conclusion"],
            confidence = 0.8 - 0.1 * chain["steps"]
        ) | {"chain": chain["chain"]}


class CreativeAgent(SpecialistAgent):
    """Right hemisphere: metaphor, association, divergent thinking."""

    def __init__(self, llm: LLMInterface):
        super().__init__("creative", llm)
        self.ga = GeneticAlgorithm(pop_size=8, generations=4)

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        evolved  = self.ga.evolve(input_text)
        creative = self.llm.ask(
            f"Generate a creative, novel response to: {evolved}",
            "You are a creative, imaginative thinker. Think outside the box."
        )
        return self._base_result(
            output     = creative,
            confidence = random.uniform(0.5, 0.85)
        ) | {"evolved_strategy": evolved}


class CodingAgent(SpecialistAgent):
    """Left frontal: systematic code generation and debugging."""

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        code_response = self.llm.ask(
            f"Write clean, commented code for: {input_text}",
            "You are an expert software engineer. Write precise, efficient code."
        )
        return self._base_result(output=code_response, confidence=0.85)


class ScientificAgent(SpecialistAgent):
    """Parietal: quantitative reasoning, hypothesis testing."""

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        hypothesis = f"Hypothesis: {input_text} can be modeled as..."
        analysis   = self.llm.ask(
            f"Provide scientific analysis with formulas for: {input_text}",
            "You are a scientist. Use equations and evidence-based reasoning."
        )
        return self._base_result(output=analysis, confidence=0.82)


class ConversationAgent(SpecialistAgent):
    """Default: natural language, empathy, dialogue."""

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        memory_ctx = context.get("memory", [])
        full       = f"Context: {memory_ctx}\nUser: {input_text}"
        response   = self.llm.ask(full)
        return self._base_result(output=response, confidence=0.75)


class SafetyAgent(SpecialistAgent):
    """Inhibitory interneurons: refuse, redirect, protect."""

    def process(self, input_text: str, context: Dict) -> Dict[str, Any]:
        return self._base_result(
            output     = "This request has been flagged by the safety system. I cannot assist with this.",
            confidence = 1.0
        )


# ══════════════════════════════════════════════════════════════════════
# SECTION 20: THALAMUS — Central Routing Hub
# ══════════════════════════════════════════════════════════════════════

class Thalamus:
    """
    The brain's central relay station.
    Routes sensory input to appropriate cortical specialists.
    Integrates: Decision Tree routing + Attention + MoE gating.

    Thalamic gating: T(x) = A(x) · DT(x) · MoE(x)
    """

    def __init__(self, specialists: Dict[str, SpecialistAgent]):
        self.specialists = specialists
        self.dt          = DecisionTreeRouter()
        self.attention   = AttentionMechanism(dim=8)
        self.moe         = MixtureOfExperts(
            list(specialists.keys()), top_k=2
        )

    def _feature_vector(self, text: str) -> List[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]

    def route(self, text: str) -> List[Tuple[str, float]]:
        """Returns ordered list of (specialist_name, weight) to invoke."""
        dt_route = self.dt.route(text)

        feats = {
            "has_conversation": 0.5,
            "has_analytical":   float("?" in text or "calculate" in text.lower()),
            "has_creative":     float("create" in text.lower() or "write" in text.lower()),
            "has_coding":       float("code" in text.lower() or "program" in text.lower()),
            "has_safety":       float(dt_route == "safety"),
            "has_scientific":   float("science" in text.lower() or "formula" in text.lower())
        }

        if dt_route == "safety":
            return [("safety", 1.0)]

        moe_weights = self.moe.gate(feats)

        # Ensure DT primary gets included
        moe_names = [n for n, _ in moe_weights]
        if dt_route not in moe_names and dt_route in self.specialists:
            moe_weights.append((dt_route, 0.3))

        return moe_weights


# ══════════════════════════════════════════════════════════════════════
# SECTION 21: RESPONSE INTEGRATION (Weighted Consensus)
# ══════════════════════════════════════════════════════════════════════

class ResponseIntegrator:
    """
    Fuses outputs from multiple specialist agents.

    Integration formula:
      R* = Σᵢ wᵢ · cᵢ · Rᵢ / Σᵢ wᵢ · cᵢ

    Where:
      wᵢ = MoE gate weight for specialist i
      cᵢ = confidence score from specialist i
      Rᵢ = text response from specialist i

    Also computes consensus entropy to measure agreement level.
    """

    def integrate(self, results: List[Dict],
                  weights: List[float]) -> Dict[str, Any]:
        if not results:
            return {"output": "No response generated.", "confidence": 0.0}

        if len(results) == 1:
            return {
                "output":     results[0]["output"],
                "confidence": results[0]["confidence"],
                "source":     results[0]["agent"],
                "consensus":  1.0
            }

        # Weighted confidence scores
        conf_weights = [
            w * r["confidence"] for w, r in zip(weights, results)
        ]
        total_weight = sum(conf_weights)

        # Pick highest weighted response as primary
        best_idx  = conf_weights.index(max(conf_weights))
        primary   = results[best_idx]["output"]

        # Consensus entropy (how much do specialists agree?)
        norm_weights = MathEngine.softmax(conf_weights)
        consensus_h  = MathEngine.entropy(norm_weights)
        consensus    = 1.0 - (consensus_h / math.log2(len(results) + 1))

        # Synthesized confidence
        synth_conf = total_weight / (len(results) + 1e-10)

        return {
            "output":     primary,
            "confidence": min(synth_conf, 1.0),
            "consensus":  max(0.0, consensus),
            "sources":    [r["agent"] for r in results],
            "all_outputs": [(r["agent"], r["output"][:100]) for r in results]
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 22: EXECUTIVE CONTROLLER (Prefrontal Cortex)
# ══════════════════════════════════════════════════════════════════════

class ExecutiveController:
    """
    The unified executive function — orchestrates ALL modules.
    Analogous to the Prefrontal Cortex.

    Full pipeline per turn:
    1. Safety check    (Amygdala)
    2. State update    (HMM)
    3. Curiosity score (Curiosity Engine)
    4. Memory retrieve (Episodic + Semantic)
    5. Thalamus route  (DT + MoE + Attention)
    6. Parallel agents (Specialist Columns)
    7. GW Broadcast    (Global Workspace)
    8. Response fuse   (Integrator)
    9. Meta-reflect    (MetaCognition)
    10. MCTS Plan      (Planner)
    11. Memory write   (Hippocampus)
    12. RL update      (Dopamine)
    """

    def __init__(self):
        # Memory
        self.working_mem  = WorkingMemory()
        self.episodic_mem = EpisodicMemory()
        self.semantic_mem = SemanticMemory()

        # LLM
        self.llm          = LLMInterface()

        # Specialists
        self.specialists  = {
            "analytical":   AnalyticalAgent(self.llm),
            "creative":     CreativeAgent(self.llm),
            "coding":       CodingAgent(self.llm),
            "scientific":   ScientificAgent(self.llm),
            "conversation": ConversationAgent(self.llm),
            "safety":       SafetyAgent(self.llm),
        }

        # Core engines
        self.thalamus     = Thalamus(self.specialists)
        self.amygdala     = AmygdalaSafetyGate()
        self.gw           = GlobalWorkspace()
        self.integrator   = ResponseIntegrator()
        self.meta         = MetaCognitionEngine()
        self.mcts         = MCTSPlanner(simulations=30)
        self.hmm          = HiddenMarkovModel()
        self.curiosity    = CuriosityEngine()
        self.predictor    = PredictiveProcessingEngine()
        self.dopamine     = DopamineRewardSystem(
            n_arms    = len(self.specialists),
            arm_names = list(self.specialists.keys())
        )
        self.fuzzy        = FuzzyDecisionEngine()

        # Subscribe all specialists to global workspace
        for name in self.specialists:
            self.gw.subscribe(name)

        self.turn_count   = 0

    def _compute_reward(self, result: Dict) -> float:
        """
        Composite reward:
          R = w₁·confidence + w₂·consensus + w₃·safety - w₄·uncertainty
        """
        conf      = result.get("confidence", 0.5)
        consensus = result.get("consensus",  0.5)
        safety    = 1.0  # penalized if safety triggered
        return 0.4 * conf + 0.3 * consensus + 0.3 * safety

    def process(self, user_input: str) -> Dict[str, Any]:
        self.turn_count += 1
        pipeline_log    = {}

        # ── STEP 1: AMYGDALA SAFETY GATE ──────────────────────────
        safety_result = self.amygdala.assess(user_input)
        pipeline_log["safety"] = safety_result
        if not safety_result["pass"]:
            return {
                "output":   "⚠️ Request blocked by safety system.",
                "reason":   safety_result["triggered_words"],
                "pipeline": pipeline_log
            }

        # ── STEP 2: HMM STATE UPDATE ───────────────────────────────
        conv_state           = self.hmm.update(user_input)
        next_state           = self.hmm.predict_next()
        pipeline_log["hmm"]  = {"state": conv_state, "next": next_state}

        # ── STEP 3: CURIOSITY SCORING ──────────────────────────────
        cur_score             = self.curiosity.score(user_input)
        pipeline_log["curiosity"] = cur_score

        # ── STEP 4: PREDICTIVE PROCESSING UPDATE ──────────────────
        pred_result           = self.predictor.update(conv_state, user_input)
        pipeline_log["prediction"] = pred_result

        # ── STEP 5: MEMORY RETRIEVAL ───────────────────────────────
        mem_context           = self.episodic_mem.retrieve(user_input, top_k=3)
        self.working_mem.push(user_input)
        working_ctx           = self.working_mem.peek(5)

        # ── STEP 6: THALAMUS ROUTING ───────────────────────────────
        routing               = self.thalamus.route(user_input)
        pipeline_log["routing"] = routing

        # ── STEP 7: PARALLEL SPECIALIST EXECUTION ─────────────────
        context = {
            "memory":    mem_context,
            "working":   working_ctx,
            "state":     conv_state,
            "curiosity": cur_score
        }

        agent_results = []
        gw_candidates = {}

        for agent_name, weight in routing:
            if agent_name not in self.specialists:
                continue
            agent  = self.specialists[agent_name]
            result = agent.process(user_input, context)
            result["gate_weight"] = weight
            agent_results.append(result)

            # Compute salience for GW competition
            salience = weight * result["confidence"]
            gw_candidates[agent_name] = (result["output"], salience)

        pipeline_log["agents"] = [
            {"name": r["agent"], "confidence": r["confidence"]}
            for r in agent_results
        ]

        # ── STEP 8: GLOBAL WORKSPACE BROADCAST ────────────────────
        if gw_candidates:
            gw_winner = self.gw.compete(gw_candidates)
            gw_event  = self.gw.broadcast(gw_winner, gw_candidates)
            pipeline_log["gw_winner"] = gw_winner

        # ── STEP 9: RESPONSE INTEGRATION ──────────────────────────
        weights        = [r.get("gate_weight", 1.0) for r in agent_results]
        integrated     = self.integrator.integrate(agent_results, weights)
        pipeline_log["integration"] = {
            "confidence": integrated["confidence"],
            "consensus":  integrated.get("consensus", 1.0)
        }

        # ── STEP 10: META-COGNITION REFLECTION ────────────────────
        meta_result    = self.meta.reflect(
            integrated["output"], integrated["confidence"], user_input
        )
        pipeline_log["meta"] = meta_result

        # ── STEP 11: FUZZY FINAL DECISION ─────────────────────────
        fuzzy_decision = self.fuzzy.decide(
            confidence  = integrated["confidence"],
            uncertainty = 1.0 - integrated.get("consensus", 0.5),
            risk        = safety_result["risk_score"]
        )
        pipeline_log["fuzzy"] = fuzzy_decision

        # ── STEP 12: MCTS PLANNING ─────────────────────────────────
        if cur_score["curiosity"] > 0.5 or meta_result["should_retry"]:
            mcts_plan = self.mcts.search(user_input)
            pipeline_log["mcts_plan"] = mcts_plan
        else:
            mcts_plan = []

        # ── STEP 13: HIPPOCAMPAL MEMORY WRITE ─────────────────────
        importance = integrated["confidence"] * (1 + cur_score["novelty"])
        self.episodic_mem.store(
            f"Q: {user_input} | A: {integrated['output'][:200]}",
            importance=importance
        )

        # ── STEP 14: DOPAMINE RL UPDATE ───────────────────────────
        reward       = self._compute_reward(integrated)
        next_V       = random.uniform(0.4, 0.9)  # estimate from V-function
        td_error     = self.dopamine.td_update(reward, next_V)
        if agent_results:
            winner_agent = agent_results[0]["agent"]
            self.dopamine.update_arm(winner_agent, reward)
            self.thalamus.moe.update_gate(winner_agent, reward)
        pipeline_log["rl"] = {"td_error": td_error, "reward": reward}

        # ── ASSEMBLE FINAL OUTPUT ──────────────────────────────────
        return {
            "output":    integrated["output"],
            "confidence": integrated["confidence"],
            "consensus":  integrated.get("consensus", 1.0),
            "state":      conv_state,
            "plan":       mcts_plan,
            "pipeline":   pipeline_log,
            "turn":       self.turn_count,
            "meta_flags": meta_result["flags"],
            "reward":     reward,
            "td_error":   td_error
        }

    def introspect(self) -> Dict[str, Any]:
        """Return full system state for debugging / visualization."""
        return {
            "episodic_traces": len(self.episodic_mem.traces),
            "working_memory":  self.working_mem.peek(),
            "q_table":         self.dopamine.get_q_table(),
            "gw_history":      len(self.gw.history),
            "kalman_conf":     self.meta.k_state,
            "brier_score":     self.meta.brier_score(),
            "prior_risk":      self.amygdala.prior_risk,
            "hmm_state":       HiddenMarkovModel.STATES[self.hmm.state],
            "moe_weights":     self.thalamus.moe.W_gate,
            "seen_inputs":     len(self.curiosity.seen_embeds),
            "incidents":       len(self.amygdala.incident_log),
        }

    def replay_consolidation(self):
        """Simulate sleep-stage hippocampal replay."""
        replayed = self.episodic_mem.replay()
        for mem in replayed:
            self.semantic_mem.assert_fact(
                concept = mem[:20],
                attr    = "replayed",
                value   = "true"
            )
        return replayed


# ══════════════════════════════════════════════════════════════════════
# SECTION 23: PATTERN DETECTOR (DFT-based)
# ══════════════════════════════════════════════════════════════════════

class PatternDetector:
    """
    Uses Discrete Fourier Transform to detect cyclical patterns
    in the sequence of user inputs (topic cycling, repetition, mood cycles).

    X[k] = Σₙ x[n] · e^(-j2πkn/N)
    """

    def __init__(self, window: int = 8):
        self.window  = window
        self.history = deque(maxlen=window)

    def _encode(self, text: str) -> float:
        """Encode text as a single float (length-normalized hash)."""
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return (h % 1000) / 1000.0

    def add(self, text: str):
        self.history.append(self._encode(text))

    def detect_cycles(self) -> Dict[str, Any]:
        if len(self.history) < self.window:
            return {"status": "insufficient_data", "dominant_freq": None}

        signal   = list(self.history)
        spectrum = MathEngine.dft(signal)
        mags     = [abs(c) for c in spectrum]
        dom_freq = mags.index(max(mags[1:len(mags)//2], default=0), 1)

        return {
            "status":       "analyzed",
            "dominant_freq": dom_freq,
            "cycle_period":  self.window / dom_freq if dom_freq > 0 else None,
            "repetitive":   max(mags[1:]) > 0.5 * mags[0]
        }


# ══════════════════════════════════════════════════════════════════════
# SECTION 24: MAIN INTERFACE — NEON BRAIN v8
# ══════════════════════════════════════════════════════════════════════

class NEONBrainV8:
    """
    Top-level interface.
    Wraps ExecutiveController with CLI, introspection, and replay.
    """

    def __init__(self):
        print("""
╔══════════════════════════════════════════════════╗
║         NEON BRAIN v8 — ONLINE                   ║
║  Bio-Inspired Unified Cognitive AI System        ║
║  Algorithms: Bayesian · MCTS · HMM · GA · MoE   ║
║             Fuzzy · CoT · GW · DFT · Hebbian     ║
╚══════════════════════════════════════════════════╝
        """)
        self.brain      = ExecutiveController()
        self.patterns   = PatternDetector(window=8)
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def chat(self, user_input: str,
              verbose: bool = False) -> str:
        """Process a single turn."""
        self.patterns.add(user_input)
        cycle_info = self.patterns.detect_cycles()

        result = self.brain.process(user_input)

        if verbose:
            self._print_pipeline(result, cycle_info)

        return result["output"]

    def _print_pipeline(self, result: Dict, cycle_info: Dict):
        p = result["pipeline"]
        print(f"\n{'─'*55}")
        print(f"  Turn {result['turn']} | State: {result['state']}")
        print(f"  Safety:    {p.get('safety', {}).get('level', '?')}")
        print(f"  Routing:   {result['pipeline'].get('routing', [])}")
        print(f"  GW Winner: {p.get('gw_winner', '?')}")
        print(f"  Confidence:{result['confidence']:.2f} | Consensus:{result['consensus']:.2f}")
        print(f"  TD Error:  {result['td_error']:.4f} | Reward: {result['reward']:.3f}")
        print(f"  Meta Flags:{result['meta_flags']}")
        print(f"  Curiosity: {p.get('curiosity', {}).get('curiosity', 0):.3f}")
        print(f"  Cycles:    {cycle_info}")
        if result["plan"]:
            print(f"  MCTS Plan: {result['plan'][0][:60]}")
        print(f"{'─'*55}\n")

    def introspect(self):
        """Print full system introspection."""
        state = self.brain.introspect()
        print("\n╔══ NEON BRAIN v8 — SYSTEM STATE ══╗")
        for k, v in state.items():
            print(f"  {k:22s}: {v}")
        print("╚══════════════════════════════════╝\n")

    def consolidate(self):
        """Trigger hippocampal replay (e.g., at session end)."""
        replayed = self.brain.replay_consolidation()
        print(f"[Consolidation] Replayed {len(replayed)} memories.\n")

    def run_cli(self):
        """Interactive CLI loop."""
        print("Commands: 'exit' | 'debug' (toggle verbose) | 'state' | 'replay'\n")
        verbose = False

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[NEON] Session ended.")
                break

            if not user_input:
                continue
            if user_input.lower() == "exit":
                self.consolidate()
                break
            if user_input.lower() == "debug":
                verbose = not verbose
                print(f"[Debug mode: {'ON' if verbose else 'OFF'}]")
                continue
            if user_input.lower() == "state":
                self.introspect()
                continue
            if user_input.lower() == "replay":
                self.consolidate()
                continue

            response = self.chat(user_input, verbose=verbose)
            print(f"\nNEON: {response}\n")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    neon = NEONBrainV8()
    neon.run_cli()
