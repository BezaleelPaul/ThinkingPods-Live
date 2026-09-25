# Design Thinking Pods (Formerly Hexapods)

## Project Documentation and Development Journey

### Team Members

* **Bezaleel Paul** – AI Development, System Architecture, Optimization, Integration
* **Madhu** – UI/UX Design, Professional Styling, Theme Development
* **Adithya S** – Hardware Research, Testing, Bug Detection, System Improvements

---

## Project Overview

The project initially began in February under the name **Hexapods**. The original vision was to create a system that could run **six independent Design Thinking stations simultaneously using a single computer system**. Each station would represent one stage of the Design Thinking process and provide AI assistance to users.

The initial estimated budget for the project was ₹50,000. However, the available budget was gradually reduced to ₹15,000 and eventually to almost nothing. Due to these financial limitations, our team had to either abandon the project or redesign it using only the resources and technology available to us. We chose the second option.

---

## Phase 1: Hardware Planning

For nearly three weeks, we evaluated various hardware options.

* **Adithya S** researched and identified suitable hardware components.
* **Bezaleel Paul** and **Madhu** analysed specifications required to support six AI stations simultaneously.

We discovered that due to rising RAM prices and hardware limitations, building a six-station AI system within a ₹15,000 budget was impractical. The required CPU and memory resources exceeded our budget constraints.

---

## Phase 2: Software Optimization Approach

To overcome hardware limitations, I explored software-based optimization methods.

I investigated concepts involving:
* Data Structures and Algorithms
* Efficient memory management
* Caching techniques
* KV Cache Quantization
* Model compression strategies

**KV Cache Quantization** reduces the precision of Key (K) and Value (V) tensors during autoregressive text generation, significantly lowering memory requirements and improving inference efficiency.

The goal was to reduce RAM consumption sufficiently to enable local AI execution on limited hardware.

---

## Phase 3: Local AI Implementation

Our entire AI pipeline was designed to run completely offline. This introduced significant computational challenges because all processing depended on local CPU and GPU resources.

We initially used:
* **Llama.cpp**
* **Qwen3-1.5B**

Qwen3-1.5B was selected because of its relatively small parameter size and high customizability. However, it had an important limitation: it behaved as a general-purpose AI and frequently attempted to answer user questions instead of guiding users through the Design Thinking process by asking critical thinking questions.

---

## Phase 4: Model Experimentation

To better suit our objectives, we shifted our approach. The project's name was changed from **Hexapods** to **Design Thinking Pods**.

The primary objective became:
> **"The AI should ask questions rather than provide answers, thereby encouraging critical thinking and idea generation."**

We experimented with:
### DeepSeek R1 Distill

* **Advantages**:
  * Strong reasoning abilities
  * Detective-like analytical questioning
  * Better contextual understanding
* **Limitations**:
  * Heavy computational requirements
  * Response times reaching approximately 200 seconds

Although reasoning quality improved, inference speed remained unsuitable for practical usage.

---

## Phase 5: Voice System Development

We experimented with multiple speech technologies.

### Kokoro TTS
* **Advantages**: High-quality speech synthesis
* **Limitations**: Significant CPU usage

### Piper TTS and Whisper
* **Advantages**: Accurate speech recognition and synthesis, lower resource requirements
* **Limitations**: The speech pipeline performed well, but the language models remained the bottleneck.

---

## Phase 6: Architecture Redesign

While researching optimization methods, I came across information regarding lightweight customizable language models and frameworks such as **ReqGPT**. This inspired me to redesign the entire system from scratch.

The new architecture included:
* **Ollama**
* **Mistral Language Model**
* **ReqGPT**
* **Llama.cpp**
* **Silero Voice Models**
* **Offline implementations** inspired by LiveKit components

I initially built the interface using Streamlit. However, graphical interfaces introduced additional overhead and latency. To improve performance, I migrated the system to a non-GUI architecture (such as the optimized FastAPI voice layer).

---

## Final Outcome

After redesigning the architecture:
* Response times improved significantly.
* Voice interaction became nearly real-time.
* The system functioned similarly to a lightweight offline version of ChatGPT Voice Mode.
* Although slight delays remained, performance became practical for real-world use.

The system achieved our original goal: **An entirely offline AI assistant capable of guiding users through the Design Thinking process using voice interaction and critical questioning.**

---

## Design and User Experience

* **Madhu** significantly contributed to:
  * Professional interface design
  * AI-inspired visual themes
  * Futuristic and Cyberpunk user interfaces
  * Overall user experience improvements
* **Adithya S** contributed through:
  * Hardware research
  * Error detection
  * Debugging
  * Feature suggestions
  * System testing and improvements

Both team members continuously provided ideas and improvements throughout development.

---

## Optimization Achievements

The project underwent major optimization:
* **Initial size**: ~7 GB
* **First reduction**: ~4 GB
* **Second reduction**: ~2.3 GB
* **Final application size**: ~15 MB (excluding AI models)
* **Estimated size with models**: ~1 GB

This represented a substantial reduction in storage requirements while simultaneously improving system performance.

---

## Current Status & Vision

The project is currently in active development. Current objectives include:
1. Improving voice understanding and speech recognition.
2. Enhancing critical questioning capabilities.
3. Refining the Design Thinking workflow.
4. Improving interface design and user experience.
5. Increasing performance and reducing latency.

### Vision Statement
**Design Thinking Pods** aims to become an accessible, low-cost, offline AI system that promotes creativity and critical thinking by guiding users through structured questioning rather than directly providing answers.

The project demonstrates that innovative AI solutions can be developed even under severe budget constraints through software optimization, efficient engineering, and collaborative teamwork.
