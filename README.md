# Example VectorDB API

## Instructions

- Copy your index.* files from your previous hands on experiment
- Install the dependencies with poetry
- Modify app.py to bind to all interfaces on a unique port
- Test out your search in the browser

## Notes

We make use of faiss-cpu here, so all inferences are done by the CPU.
For your use case - millions of embeddings - you'd probably want to use GPU inference.

