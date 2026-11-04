import warnings
# Ignora avisos chatos no terminal
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Carregando o modelo de Inteligência Artificial...")
print("(Isso pode demorar um pouco na primeira vez, pois ele fará o download do modelo)\n")

# Carrega o modelo (leve e excelente para português)
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Nossas "tabelas" fictícias baseadas no seu sistema
tabelas = {
    "usuario_usuario": "Pessoas, clientes, pacientes, usuários, alunos, médicos, participantes",
    "registro_refeicao_registrorefeicao": "Pratos consumidos, alimentação, comida, refeições, dietas, o que comeu, carboidratos, pão, café",
    "dado_clinico_dadoclinico": "Glicemia, peso, altura, exames de sangue, pressão arterial, insulina, hiperglicemia"
}

nomes_tabelas = list(tabelas.keys())
descricoes = list(tabelas.values())

# Transforma as descrições das tabelas em vetores matemáticos
vetores_tabelas = modelo.encode(descricoes)

print("="*50)
print(" SISTEMA PRONTO! TESTE DE SINÔNIMOS E CONTEXTO ")
print(" Digite 'sair' para encerrar o programa.")
print("="*50)

while True:
    # Pega a frase que você digitar no terminal
    pergunta = input("\nDigite uma frase (ex: 'paciente comendo pão'): ")
    
    if pergunta.lower().strip() == 'sair':
        print("Encerrando o teste...")
        break
        
    if not pergunta.strip():
        continue

    # Transforma a sua frase em vetor
    vetor_pergunta = modelo.encode([pergunta])
    
    # Calcula a similaridade (de 0 a 1) entre a sua frase e as tabelas
    similaridades = cosine_similarity(vetor_pergunta, vetores_tabelas)[0]
    
    # Pega a maior pontuação
    indice_maior = np.argmax(similaridades)
    tabela_vencedora = nomes_tabelas[indice_maior]
    confianca = similaridades[indice_maior] * 100
    
    print("-" * 40)
    print(f"🤖 Entendi! Você está falando da tabela:")
    print(f"✅ {tabela_vencedora}")
    print(f"📊 Nível de confiança: {confianca:.1f}%")
    print("-" * 40)