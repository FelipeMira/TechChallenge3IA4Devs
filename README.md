```markdown
# Fine Tuning com Llama para Descrição de Títulos

Este projeto demonstra como realizar o ajuste fino de um modelo Llama para gerar descrições de títulos com base no conteúdo usando a biblioteca Unsloth.

## Visão Geral do Projeto

O projeto consiste nas seguintes etapas:

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
2. Instale as bibliotecas necessárias:

```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install transformers datasets
```

3. Prepare os dados de entrada e saída:

```python
import json
import re
import html
import unicodedata


# Define uma função para limpar o texto
def clean_text(text):
    # Remove caracteres indesejados e espaços extras
    text = re.sub(r'\s+', ' ', text)  # Substitui múltiplos espaços por um único espaço
    text = text.strip()  # Remove espaços extras no início e no final do texto
    text = html.unescape(text)  # Converte entidades HTML para caracteres normais
    # Ajusta os caracteres Unicode
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.encode('unicode_escape').decode('unicode_escape')
    return text

# Define uma função para preparar os prompts
def prepare_prompts(file_path, output_path):
    # Abre o arquivo JSON de entrada para leitura
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Lê todas as linhas do arquivo

    processed_data = []  # Inicializa uma lista para armazenar os dados processados

    # Itera sobre cada linha do arquivo
    for line in lines[:50000]:
        item = json.loads(line)  # Converte a linha de texto JSON em um dicionário Python
        title = clean_text(item.get('title', '')).strip()
        content = clean_text(item.get('content', '')).strip()

        # Verifica se o título e o conteúdo não estão vazios
        if title and content and title != content:
            # Formata o prompt com o título e o conteúdo
            prompt = f"DESCRIBE THE TITLE BASED ON THE CONTENT\n[|Title|] {title}[|eTitle|]\n\n[|Content|]{content}[|eContent|]"
            processed_data.append({"input": prompt})  # Adiciona o prompt formatado à lista de dados processados

    # Abre o arquivo JSON de saída para escrita
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # Salva os dados processados no arquivo JSON de saída
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

# Caminho para o arquivo JSON de entrada e o arquivo JSON de saída
input_file_path = './drive/MyDrive/FIAP/data/trn.json'
output_file_path = './drive/MyDrive/FIAP/processed/processed_prompts.json'

# Chama a função para preparar os prompts
prepare_prompts(input_file_path, output_file_path)

# Imprime uma mensagem indicando que os prompts foram salvos
print(f"Prompts preparados foram salvos em '{output_file_path}'.")
```

O código realiza a limpeza de texto e a preparação de prompts a partir de um arquivo JSON de entrada. Aqui está uma explicação geral do que cada parte do código faz:

### Função `clean_text`
Esta função limpa e normaliza o texto de entrada:
- Remove múltiplos espaços e espaços extras no início e no final do texto.
- Converte entidades HTML para caracteres normais.
- Normaliza caracteres Unicode, removendo acentos e caracteres especiais.

### Função `prepare_prompts`
Esta função prepara os prompts a partir de um arquivo JSON de entrada e salva os dados processados em um arquivo JSON de saída:
- Lê o arquivo JSON de entrada.
- Inicializa uma lista para armazenar os dados processados.
- Itera sobre cada linha do arquivo, convertendo-a em um dicionário Python.
- Limpa e verifica se o título e o conteúdo não estão vazios.
- Formata o prompt com o título e o conteúdo e adiciona à lista de dados processados.
- Salva os dados processados no arquivo JSON de saída.

### Execução do Código
Define os caminhos para o arquivo JSON de entrada e o arquivo JSON de saída, chama a função `prepare_prompts` e imprime uma mensagem indicando que os prompts foram salvos.

<br/>

4. Formate o conjunto de dados:

```python
DATA_PATH = "./drive/MyDrive/FIAP/processed/processed_prompts.json"
OUTPUT_PATH_DATASET = "./drive/MyDrive/FIAP/processed/processed_prompts_dataset.json"
max_seq_length = 2048
dtype = None
load_in_4bit = True
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
]
    
def format_dataset_into_model_input(data):
    def separate_text(full_text):
        title_start = full_text.find("[|Title|]") + len("[|Title|]")
        title_end = full_text.find("[|eTitle|]")
        content_start = full_text.find("[|Content|]") + len("[|Content|]")
        content_end = full_text.find("[|eContent|]")

        instruction = full_text.split('\n')[0]
        input_text = full_text[title_start:title_end].strip()
        response = full_text[content_start:content_end].strip()

        return instruction, input_text, response

    # Inicializando as listas para armazenar os dados
    instructions = []
    inputs = []
    outputs = []

    # Processando o dataset
    for item in data:  # Iterar diretamente sobre a lista de dicionários
        instruction, input_text, response = separate_text(item['input'])
        instructions.append(instruction)
        inputs.append(input_text)
        outputs.append(response)

    # Criando o dicionário final
    formatted_data = {
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    }

    # Salvando o resultado em um arquivo JSON
    with open(OUTPUT_PATH_DATASET, 'w') as output_file:
        json.dump(formatted_data, output_file, indent=4)

    print(f"Dataset salvo em {OUTPUT_PATH_DATASET}")
```

5. Carregue e adapte o modelo:

```python
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

from unsloth import FastLanguageModel, is_bfloat16_supported

format_dataset_into_model_input(data)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",

    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

Este trecho de código está aplicando uma técnica chamada PEFT (Parameter-Efficient Fine-Tuning) ao modelo de linguagem `model` previamente carregado. O objetivo do PEFT é ajustar o modelo para uma tarefa específica de forma eficiente, modificando apenas uma pequena parte dos parâmetros do modelo original. Isso economiza tempo e recursos computacionais em comparação com o ajuste fino tradicional, que treina todos os parâmetros.

Vamos entender os parâmetros da função `FastLanguageModel.get_peft_model`:

- model: O modelo de linguagem base que será ajustado.

- r = 16: Define o rank da matriz LoRA (Low-Rank Adaptation), que é a técnica principal usada pelo PEFT. Um rank menor significa menos parâmetros a serem ajustados, tornando o processo mais eficiente.

- target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]: Especifica os módulos do modelo base onde as matrizes LoRA serão aplicadas. Esses módulos geralmente são partes importantes da arquitetura do modelo, como camadas de atenção.

- lora_alpha = 16: Um fator de escala que controla a influência das matrizes LoRA no processo de ajuste fino.

- lora_dropout = 0: A taxa de dropout aplicada às matrizes LoRA. Dropout é uma técnica de regularização que ajuda a evitar o overfitting.

- bias = "none": Indica que nenhum bias (termo de polarização) será adicionado às matrizes LoRA.

- use_gradient_checkpointing = "unsloth": Habilita uma técnica chamada gradient checkpointing, que reduz o consumo de memória durante o treinamento. O valor "unsloth" provavelmente se refere a uma implementação específica dessa técnica pela biblioteca Unsloth.

- random_state = 3407: Define a semente para o gerador de números aleatórios, garantindo a reprodutibilidade dos resultados.

- use_rslora = False: Desabilita o uso de uma variante do LoRA chamada "Rotary Sparse LoRA" (RSLoRA).

- loftq_config = None: Desabilita o uso de outra técnica de quantização chamada "Low-bit Optimization of Quantized Transformers" (LOFTQ).

**Em resumo, este código está preparando o modelo de linguagem para um ajuste fino eficiente utilizando a técnica PEFT com LoRA. Ele configura os parâmetros da adaptação, especificando onde as modificações serão aplicadas e como elas influenciarão o treinamento.** O resultado final é um modelo ajustado para a tarefa desejada, com um menor custo computacional em comparação ao ajuste fino tradicional.

6. Teste o modelo antes do treinamento:

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):

        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset

dataset = load_dataset("json", data_files=OUTPUT_PATH_DATASET, split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

def test_model_alpaca(instruction, input_text=""):
    global model
    # Preparar o modelo para inferência
    model = FastLanguageModel.for_inference(model)

    # Formatar o prompt no formato Alpaca
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    # Tokenizar o prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device="cuda")

    # Gerar a saída pelo modelo
    outputs = model.generate(**inputs)

    # Decodificar a saída
    predicted_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extrair a resposta após "### Response:"
    predicted_response = predicted_response.split("### Response:")[-1].strip()

    return predicted_response

title_question_example = "The GOOD LUCK PENCIL"  # Exemplo de título como pergunta

predicted_content = test_model_alpaca(title_question_example)  # Chama a função com o título
print(f"Título/Pergunta: {title_question_example}")
print(f"Conteúdo:\n{predicted_content}")  # Imprime o conteúdo
```

### Resposta do teste

Título/Pergunta: The GOOD LUCK PENCIL
Conteúdo:
The GOOD LUCK PENCIL

### Hint:
The GOOD LUCK PENCIL

### Solution:
The GOOD LUCK PENCIL

7. Treine o modelo:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="none",
    ),
)

import torch

torch.cuda.empty_cache()

import gc
gc.collect()

trainer_stats = trainer.train()
```

O código cria um objeto `SFTTrainer` para treinar um modelo de linguagem com parâmetros específicos. Aqui está uma explicação detalhada:

- `model`: O modelo de linguagem a ser treinado.
- `tokenizer`: O tokenizador usado para processar o texto.
- `train_dataset`: O conjunto de dados de treinamento.
- `dataset_text_field`: O campo de texto no conjunto de dados.
- `max_seq_length`: O comprimento máximo da sequência de entrada.
- `dataset_num_proc`: Número de processos para carregar o conjunto de dados.
- `packing`: Se `True`, empacota várias sequências em uma única entrada.
- `args`: Argumentos de treinamento fornecidos pelo `TrainingArguments`.

Dentro de `TrainingArguments`:
- `per_device_train_batch_size`: Tamanho do lote por dispositivo.
- `gradient_accumulation_steps`: Número de passos de acumulação de gradiente.
- `warmup_steps`: Número de passos de aquecimento.
- `max_steps`: Número máximo de passos de treinamento.
- `learning_rate`: Taxa de aprendizado.
- `fp16`: Usa precisão de 16 bits se `bfloat16` não for suportado.
- `bf16`: Usa `bfloat16` se suportado.
- `logging_steps`: Frequência de registro de logs.
- `optim`: Otimizador usado (`adamw_8bit`).
- `weight_decay`: Decaimento de peso.
- `lr_scheduler_type`: Tipo de agendador de taxa de aprendizado (`linear`).
- `seed`: Semente para reprodutibilidade.
- `output_dir`: Diretório de saída para salvar os resultados.
- `report_to`: Destino dos relatórios de treinamento (`none`).

### Resposta do Treinamento

[60/60 04:11, Epoch 0/1]
| Step | Training Loss |
|------|---------------|
| 1    | 2.661300      |
| 2    | 2.684400      |
| 3    | 2.560900      |
| 4    | 2.778100      |
| 5    | 2.821200      |
| 6    | 2.454100      |
| 7    | 2.301400      |
| 8    | 2.253100      |
| 9    | 2.049500      |
| 10   | 2.302500      |
| 11   | 2.139600      |
| 12   | 1.845200      |
| 13   | 1.640500      |
| 14   | 1.987300      |
| 15   | 2.026600      |
| 16   | 1.870900      |
| 17   | 2.133700      |
| 18   | 1.779300      |
| 19   | 1.768000      |
| 20   | 1.592600      |
| 21   | 1.713600      |
| 22   | 1.721500      |
| 23   | 1.948300      |
| 24   | 1.870500      |
| 25   | 2.102800      |
| 26   | 1.796900      |
| 27   | 1.916000      |
| 28   | 1.645500      |
| 29   | 2.040200      |
| 30   | 1.602100      |
| 31   | 1.898200      |
| 32   | 1.857300      |
| 33   | 1.798700      |
| 34   | 1.849200      |
| 35   | 1.531400      |
| 36   | 1.814300      |
| 37   | 1.933300      |
| 38   | 1.582500      |
| 39   | 1.910600      |
| 40   | 2.017200      |
| 41   | 2.067600      |
| 42   | 1.818500      |
| 43   | 2.032600      |
| 44   | 1.848600      |
| 45   | 1.629500      |
| 46   | 1.730000      |
| 47   | 1.855700      |
| 48   | 1.773400      |
| 49   | 1.624300      |
| 50   | 1.975400      |
| 51   | 2.072700      |
| 52   | 2.003800      |
| 53   | 2.080400      |
| 54   | 1.971600      |
| 55   | 1.966600      |
| 56   | 1.775000      |
| 57   | 1.691300      |
| 58   | 1.980800      |
| 59   | 1.448700      |
| 60   | 1.853500      |


8. Realize a inferência:

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format("DESCRIBE THE TITLE BASED ON THE CONTENT.",
        "The GOOD LUCK PENCIL",
        "",
  )
], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

- FastLanguageModel.for_inference(model):

  Esta linha prepara o modelo model para inferência, ou seja, para gerar texto.
Internamente, isso pode envolver otimizações para tornar a geração de texto mais rápida e eficiente, como desabilitar o dropout ou outras técnicas de regularização usadas durante o treinamento.

- inputs = tokenizer([...], return_tensors="pt").to("cuda"):

  Esta linha tokeniza o prompt de entrada e o converte em um formato adequado para o modelo.
tokenizer(...): O tokenizador divide o prompt em unidades menores (tokens) que o modelo consegue entender.
alpaca_prompt.format(...): Formata o prompt no estilo Alpaca, que inclui instruções, entrada e (neste caso, vazia) saída esperada. O prompt específico solicita ao modelo que "DESCREVA O TÍTULO COM BASE NO CONTEÚDO" para o título "The GOOD LUCK PENCIL".
return_tensors="pt": Especifica que os tokens devem ser retornados como tensores PyTorch.
.to("cuda"): Move os tensores para a GPU (se disponível) para acelerar o processamento.

-  `outputs = model.generate(inputs, max_new_tokens=64, use_cache=True)`:**

  Esta linha usa o modelo para gerar texto com base no prompt tokenizado.
model.generate(...): Chama a função de geração de texto do modelo.
**inputs: Passa os tokens do prompt como entrada para o modelo.
max_new_tokens=64: Limita o número máximo de tokens gerados pelo modelo a 64.
use_cache=True: Habilita o cache para acelerar a geração de texto, reutilizando cálculos anteriores.

-  tokenizer.batch_decode(outputs):

  Esta linha decodifica os tokens gerados pelo modelo de volta para texto legível.
tokenizer.batch_decode(...): Converte os tokens de saída em texto.


## Resultado Obtido

Gr. 2-4--Esta é a história de um lápis que foi usado por várias crianças e agora foi passado para uma nova dona, uma garotinha chamada Mary. O lápis foi repassado por diversas crianças e cada uma acrescentou suas próprias palavras à história.

História do dataset (Inglês):

Mary Ann can't do anything wrong especially after finding a good-luck pencil. She works all the math problems right and draws pictures that everyone admires. When her teacher assigns a homework composition, Mary Ann must find something interesting to write about her ordinary family, so she lets the good-luck pencil go to work. It writes that her mother is a world-famous ballerina, her father is an astronaut and Mary Ann a world champion piano player planning to climb Mt. Olympus in the summer. But then the composition becomes real. After practicing Mozart and Bach over and over, then doing hours of exhausting exercises, Mary Ann longs for a regular life again. She pulls out a different pencil and writes a less exciting but accurate composition, dispelling the unwanted illusion. The illustrations are lighthearted and will draw young readers into repeated readings. Copyright 1986 Reed Business Information, Inc.

(Português):

Mary Ann não pode fazer nada de errado, especialmente depois de encontrar um lápis da sorte. Ela resolve todos os problemas de matemática corretamente e faz desenhos que todos admiram. Quando sua professora atribui uma redação de lição de casa, Mary Ann deve encontrar algo interessante para escrever sobre seu cotidiano a família, então ela deixa o lápis da sorte trabalhar. Ele escreve que sua mãe é uma bailarina mundialmente famosa, seu pai é um astronauta e Mary Ann uma pianista campeã mundial que planeja escalar o Monte. então a composição se torna real Depois de praticar Mozart e Bach repetidamente e depois fazer horas de exercícios exaustivos, Mary Ann anseia por uma vida normal novamente. Ela pega um lápis diferente e escreve uma composição menos emocionante, mas precisa, dissipando a ilusão indesejada. alegre e atrairá jovens leitores para leituras repetidas. Copyright 1986 Reed Business Information, Inc.
```
