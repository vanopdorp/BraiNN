# BraiNN  
An Experimental Neural Architecture with Working Memory, Relational Reasoning, and Adaptive Learning

BraiNN is a compact research‑oriented neural network that blends multiple cognitive‑inspired components into a single architecture. It is designed to move beyond standard language modeling by integrating mechanisms for memory, relational structure, concept extraction, and fast online adaptation. The goal is to explore models that not only generate text, but also **store**, **organize**, **reason**, and **learn** during interaction.

---

## Core Capabilities

### Working Memory
A differentiable memory module with multiple slots, soft addressing, gated writing, and persistent state. It allows the model to maintain information across steps, enabling multi‑step reasoning and contextual continuity.

### Relational World Model
A lightweight graph‑based system that stores subject–action–object triples extracted from hidden states. Nodes decay over time, and the model retrieves relational context using similarity‑based querying. This provides a structured knowledge layer that evolves dynamically.

### Concept Extraction
A learned mechanism that separates hidden representations into **subject**, **action**, and **object** components using attention‑like weighting. These concepts feed into the relational memory and working memory.

### Liquid Self‑Attention
A recurrent, RWKV‑style attention mechanism with linear complexity. It replaces quadratic attention and supports long sequences efficiently.

### S4D State‑Space Module
A compact state‑space layer that captures long‑range temporal patterns and complements the attention and GRU components.

### MirrorLM for Fast Learning
A secondary lightweight model that tracks prediction errors and adapts rapidly. It supports meta‑learning signals, fast‑weights, and online updates without destabilizing the main model.

### Hippocampus Memory Buffer
A prioritized replay buffer that stores surprising or important sentences. It supports consolidation, continual learning, and long‑term retention.

---

## Architecture Overview

- **DynamicTokenizer**  
  Hybrid character/subword tokenizer that grows with new observations.

- **LiquidLM**  
  The main model combining embeddings, liquid attention, S4D blocks, concept extraction, relational memory, working memory, and a language‑modeling head.

- **MirrorLM**  
  A fast‑adapting auxiliary model with meta‑learning signals and fast‑weight updates.

- **RelationalWorldModel**  
  A dynamic graph of concept nodes with decay, storage, and retrieval.

- **WorkingMemory**  
  Slot‑based differentiable memory with read/write operations.

- **Hippocampus**  
  Episodic memory with priority‑based sampling.

---

## Training Workflow

- Tokenize and encode dialogue data  
- Train LiquidLM with AdamW and cosine annealing  
- Periodically evaluate on held‑out sequences  
- Generate text using autoregressive sampling  
- Export working memory, vocabulary, and hippocampus state  
- Optionally consolidate memory or perform online learning with MirrorLM

---

## Example Applications

- Compact language models with persistent memory  
- Continual learning and online adaptation experiments  
- Research on relational reasoning and concept extraction  
- Cognitive‑inspired architectures and hybrid neural systems  
- Small‑scale LM research without massive compute requirements

---

## Requirements

- Python 3  
- PyTorch  
- CUDA (optional but recommended)

---

## Project Status

BraiNN is an active experimental project.  
Expect rapid iteration, architectural changes, and ongoing exploration of memory‑augmented neural systems.

