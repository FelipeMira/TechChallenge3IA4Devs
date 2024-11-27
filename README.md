# Fine Tuning com Llama para Descrição de Títulos

Este projeto demonstra como realizar o ajuste fino de um modelo Llama para gerar descrições de títulos com base no conteúdo usando a biblioteca Unsloth.

## Visão Geral do Projeto

O projeto consiste nas seguintes etapas:

0. **Download o dataset da Amazon:**
   - Download dataset
   - Mount um drive no google
   - Cria as pastas necessárias
   - Unzip e descompacta na pasta alvo

1. **Preparação de Dados:**
   - Limpa e pré-processa dados de texto de um arquivo JSON.
   - Formata os dados em prompts adequados para ajuste fino.

2. **Tokenização e Codificação:**
   - Utiliza a biblioteca `unsloth` para tokenizar e codificar os dados de texto para o treinamento do modelo.
   - Requer um ambiente de execução com GPU para desempenho ideal.

3. **Carregamento e Adaptação do Modelo:**
   - Carrega um modelo Llama pré-treinado (por exemplo, llama-3-8b-bnb-4bit).
   - Aplica Ajuste Fino Eficiente em Parâmetros (PEFT) usando LoRA (Adaptação de Baixo Rank) para adaptar o modelo à tarefa específica.

4. **Formatação do Conjunto de Dados:**
   - Formata o conjunto de dados preparado no formato Alpaca, comumente usado para tarefas de instruções.
   
5. **Teste antes do treinamento:**
   - Fornece uma função (`test_model_alpaca`) para testar o modelo ajustado, gerando descrições para títulos fornecidos.
	
6. **Treinamento do Modelo:**
   - Usa o `SFTTrainer` da biblioteca `trl` para realizar o ajuste fino do modelo no conjunto de dados preparado.
   - Configura parâmetros de treinamento como tamanho do lote, taxa de aprendizado e estratégia de otimização.

7. **Inferência do Modelo:**
   - Gera texto usando o modelo com um limite de 64 tokens e, por fim, decodifica os tokens gerados em texto legível. O resultado seria uma descrição do título gerada pelo modelo, com base no seu conhecimento prévio e no formato do prompt.

## Uso

1. Certifique-se de ter um ambiente Google Colab com um ambiente de execução de GPU habilitado.
