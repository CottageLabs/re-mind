- [x] Support multi llm model (local hugging, openai...)
- [ ] Support different embedding model for vector store
- [r] Add cli tool to load file to vector store
- [ ] Support prompt template as command
- [x] Use LangGraph for complex workflow
- [ ] Define metric to evaluate the performance of RAG application
- [ ] Context length management for long documents

## Pipeline
- [ ] KTODO add plan before retrieve
- [ ] KTODO add re-rank after retrieve
- [ ] add node to extract query from user input
- [ ] add node to generate more query from base query
- [ ] try different embedding model, e.g. word, sentence, document embedding

## Human interface

- [ ] Cli interface with click + rich
- [ ] Web UI with flask + react



## Remin Chat
- [ ] Support sequences messages context
- [ ] Design how to filter with metadata
- [ ] try langgraph to call different tasks
- [ ] build general tool model with openai function calling
- [x] support search only mode 
- [x] support switch llm
- [x] output_mode [debug / detail / simple]

### Vector Store
- [ ] Make it easier to switch vector store
- [ ] Add command that show report of vector store (e.g. number of docs, size, embedding model)
- [ ] Add command to show and switch vector store
