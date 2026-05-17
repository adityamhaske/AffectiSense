# Contributing to NeuroSense AI

Thank you for your interest in contributing to NeuroSense AI! We welcome community contributions to help build a world-class, modality-resilient mental health assessment platform.

By participating in this project, you agree to abide by our Code of Conduct.

---

## 🚀 How Can I Contribute?

### 1. Reporting Bugs
* Check the current Issues list to ensure the bug hasn't already been reported.
* Open a new Issue with a clear description, steps to reproduce, and any error logs or screenshots.

### 2. Suggesting Enhancements
* Check the existing issues and roadmap to ensure the feature isn't already planned.
* Open an issue outlining the user story, proposed technical design, and clinical value.

### 3. Submitting Code Changes (Pull Requests)
* Fork the repository and create your branch from `main`.
* Keep your commits focused, small, and descriptive.
* Follow the style guidelines of the project (PEP 8 for Python, ESLint/Prettier for Next.js).
* Open a Pull Request targeting `main` with a thorough description of your changes.

---

## 🛠 Local Development Setup

### Python Backend
1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/NeuroSense-AI.git
   cd NeuroSense-AI
   ```
2. Set up virtual environment and install backend development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn backend.app.api.main:app --reload
   ```

---

## 📝 Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
* `feat:` A new feature.
* `fix:` A bug fix.
* `docs:` Documentation-only changes.
* `style:` Changes that do not affect the meaning of the code (white-space, formatting, etc.).
* `refactor:` A code change that neither fixes a bug nor adds a feature.
* `perf:` A code change that improves performance.
* `test:` Adding missing tests or correcting existing tests.

Example:
```
feat: add MediaPipe landmark processing pipeline for video modality
```

---

## ⚖️ License
By contributing to NeuroSense AI, you agree that your contributions will be licensed under the project's **MIT License**.
