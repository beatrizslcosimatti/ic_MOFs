import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. CONFIGURAÇÃO ---
# Coloque aqui o nome exato do seu arquivo (pode ser .csv ou .xlsx)
nome_do_arquivo = 'tabela_binaria_mof.xlsx' 

# --- 2. FUNÇÃO DE LIMPEZA (Corrige erros do Excel) ---
def limpar_valor_numerico(val):
    """
    Tenta converter textos e datas estranhas (ex: 2025-02-03) 
    de volta para números ou zero.
    """
    val_str = str(val).strip()
    
    # Se já for número, retorna
    if isinstance(val, (int, float)):
        return val
        
    # Corrige erros de data comuns no Excel (ex: 3.2 virou 3-fev)
    # Se encontrar formato de data ou texto estranho, tenta limpar
    try:
        # Remove caracteres que não são números (exceto ponto e traço)
        return float(val_str)
    except ValueError:
        pass
    
    # Se falhar, retorna 0 (assume dado faltante)
    return 0.0

# --- 3. CARREGAR E PREPARAR DADOS ---
print("Carregando arquivo...")

try:
    if nome_do_arquivo.endswith('.csv'):
        df = pd.read_csv(nome_do_arquivo)
    else:
        df = pd.read_excel(nome_do_arquivo)
except FileNotFoundError:
    print(f"ERRO: Não encontrei o arquivo '{nome_do_arquivo}'. Verifique o nome.")
    exit()

# Aplicar a limpeza nas colunas numéricas principais
colunas_para_limpar = [
    'Densidade de Corrente (mA/cm²)', 
    'Coef. de Difusão (cm²/s)', 
    'Constante Cinética (kₒ, M⁻¹s⁻¹)'
]

for col in colunas_para_limpar:
    if col in df.columns:
        df[col] = df[col].apply(limpar_valor_numerico)

# Preencher vazios com 0
df = df.fillna(0)

# Definir o alvo (o que queremos prever)
target = 'Densidade de Corrente (mA/cm²)'

if target not in df.columns:
    print(f"ERRO: A coluna '{target}' não existe na tabela.")
    exit()

# Separar dados (X = características, y = alvo)
X = df.drop(columns=[target])
y = df[target]

# --- 4. INTELIGÊNCIA ARTIFICIAL (Random Forest) ---
print("Treinando o modelo...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

# --- 5. RESULTADOS ---
# Calcular importância das variáveis
importancias = modelo.feature_importances_
df_importancia = pd.DataFrame({
    'Característica': X.columns,
    'Importância (%)': importancias * 100
}).sort_values(by='Importância (%)', ascending=False)

print("\n" + "="*40)
print(" RESULTADO: O QUE É MAIS IMPORTANTE? ")
print("="*40)
print(df_importancia.head(10).to_string(index=False))

# Salvar resultados em um novo Excel
df['Predição_IA'] = modelo.predict(X)
df.to_excel('resultado_final_com_predicao.xlsx', index=False)
print("\nArquivo salvo: 'resultado_final_com_predicao.xlsx'")
