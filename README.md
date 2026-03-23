# BraiNN
A Neural Network with Working Memory, Relational Reasoning, and Adaptive Learning

BraiNN is an experimental neural architecture designed to go beyond standard language models.  
It integrates working memory, relational reasoning, concept extraction, and online learning into a compact and efficient system.  
The goal is to create a model that not only predicts text, but also **remembers**, **adapts**, and **understands structure** in a more human‑like way.

---

## Features

### **• RWKV‑style LiquidSelfAttention**
A recurrent attention mechanism that scales linearly with sequence length.  
No quadratic memory usage, no large attention matrices, and stable performance on long sequences.

### **• Multi‑Layer GRU Backbone**
A stacked GRU encoder that processes token sequences and produces robust hidden representations.

### **• S4D State-Space Module**
A lightweight state‑space model that captures long‑range temporal structure and complements the GRU.

### **• Working Memory Module**
A differentiable memory system with:
- multiple memory slots  
- soft addressing  
- gated writing  
- persistent state across steps  

This allows BraiNN to store and retrieve information beyond the immediate context window.

### **• ConceptExtractor**
Extracts subject, action, and object representations from token sequences using learned attention weights.

### **• RelationalWorldModel**
Builds a dynamic graph of entities and relations:
- nodes represent concepts  
- edges represent learned relationships  
- message passing refines relational context  

This enables BraiNN to accumulate structured knowledge over time.

### **• Confidence Network**
Predicts the model’s confidence in its own output, enabling:
- uncertainty estimation  
- adaptive learning  
- selective memory updates  

### **• Hippocampus Memory Buffer**
Stores high‑priority sentences for replay and consolidation, inspired by biological memory systems.

### **• Online Learning Support**
The model can learn new sentences on the fly, expand its vocabulary, and update its internal memory without full retraining.

---

## 📦 Components Overview

- **DynamicTokenizer**  
  Builds a hybrid character/subword/word vocabulary that grows during training.

- **LiquidSelfAttention**  
  RWKV‑style recurrent attention with time‑mixing and gating.

- **WorkingMemory**  
  Differentiable memory with soft attention and gated writes.

- **RelationalWorldModel**  
  Graph‑based relational reasoning with message passing.

- **ConceptExtractor**  
  Learns to extract semantic roles from sequences.

- **LiquidLM**  
  The main language model combining all modules.

- **MirrorLM**  
  A secondary model used for meta‑learning and stability checks.

---

## Training

BraiNN use:
- curriculum learning  
- online learning  
- memory consolidation  
- adaptive learning rates  
- GPU or CPU training  

Training is performed in phases, gradually increasing linguistic complexity.

---

## Example Use Cases

- small language models with memory  
- continual learning experiments  
- reasoning over entities and relations   
- research on compact LM designs  

---

## Requirements

- Python 3 
- PyTorch  
- CUDA (optional)  

---

## License

MIT License (or your preferred license)

---

## Status

BraiNN is an active research project.  
Expect rapid changes, experimental features, and ongoing improvements.
