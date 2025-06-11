## Projeto "Cinco Minutos com Jesus" - Hora Luterana

*Sobre o Projeto*
O Cinco Minutos com Jesus é um assistente virtual inteligente desenvolvido para a Hora Luterana, especializado em temas bíblicos, citações e meditações cristãs no contexto da Igreja Luterana. Este projeto utiliza tecnologias de ponta em processamento de linguagem natural para oferecer respostas precisas e contextualizadas aos usuários.

*Funcionalidades Principais*
Respostas baseadas em documentos oficiais da Hora Luterana
Citações precisas de autores e datas quando disponíveis
Interação natural com capacidade de manter contexto da conversa
Controle de qualidade com logs detalhados de todas as interações
Sistema híbrido que combina recuperação de informação com geração de texto

*Arquitetura Técnica*
O sistema é composto por três componentes principais:

1. Banco de Dados de Vetores (Vector Store)
Utiliza FAISS para armazenamento eficiente de embeddings
Processa documentos JSON contendo meditações e reflexões
Divide textos em chunks otimizados para recuperação

2. Sistema de Recuperação (Retriever)
Gera queries otimizadas a partir das perguntas dos usuários
Avalia relevância das perguntas para temas luteranos
Recupera os documentos mais relevantes para cada consulta

3. Modelo de Linguagem (LLM)
GPT-4 para respostas finais (quando há contexto relevante)
GPT-3.5 para geração de queries de busca
Sistema rigoroso de prompts para garantir precisão teológica

*Base de Conhecimento*
O sistema utiliza uma extensa coleção de documentos da Hora Luterana, incluindo:
Meditações diárias
Estudos bíblicos
Reflexões teológicas
Comentários sobre textos litúrgicos

## Stack Tecnológica
- **Backend**: Node.js + Express
- **Banco**: Redis (conversas) + FAISS (vector store)
- **IA**: LangChain + OpenAI (GPT-4/GPT-3.5)
- **Embeddings**: text-embedding-3-large

---
*© 2025 Optimus Data Technology - Licença Acadêmica (ver [LICENSE](LICENSE))*

## Fluxo de Inteligência
```mermaid
graph TD
  B[Análise do Tema da Pergunta] -->|Tema Luterano| C[Geração de Query Otimizada]
  B -->|Tema Não Relevante| D[Resposta Direta]
  C --> E[Recuperação de Documentos]
  E --> F[Geração de Resposta Contextualizada]
  D --> G[Resposta Padrão]
  F --> H[Resposta ao Usuário]
  G --> H
  H --> I[Salvar Interação no Redis]
