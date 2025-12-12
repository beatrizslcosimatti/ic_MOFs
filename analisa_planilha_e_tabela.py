import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor

# --- 1. CONFIGURAÇÃO ---
arquivo_original = 'planilha.xlsx' 

print(f"Lendo o arquivo original: {arquivo_original}...")

try:
    if arquivo_original.endswith('.csv'):
        df = pd.read_csv(arquivo_original)
    else:
        df = pd.read_excel(arquivo_original)
except FileNotFoundError:
    print("ERRO: Arquivo não encontrado. Verifique se o nome está correto.")
    exit()

print(f"Arquivo carregado com {len(df)} linhas.")

# --- 2. LIMPEZA E CONVERSÃO ---

def limpar_numero(val):
    val = str(val).strip()
    if '2025-' in val or '4324-' in val: return 0.0
    try:
        return float(val.replace(',', '.'))
    except:
        match = re.search(r'[-+]?\d*\.\d+|\d+', val.replace(',', '.'))
        return float(match.group()) if match else 0.0

cols_num = ['Densidade de Corrente (mA/cm²)', 'Potencial de Oxidação (Ep, V vs Hg/HgO)', 
            'Área Superficial (m²/g)', 'Concentração_Etanol']

for col in cols_num:
    if col in df.columns:
        df[col] = df[col].apply(limpar_numero)

df = df.fillna(0)

# --- CRIAÇÃO E SALVAMENTO DA TABELA BINÁRIA ---
cats = ['Metal', 'Ligante', 'Metodo_Sintese', 'Tipo_Eletrolito', 'Eletrodo Catalítico']
cats_existentes = [c for c in cats if c in df.columns]

# Cria a tabela binária
df_binaria = pd.get_dummies(df, columns=cats_existentes, prefix=cats_existentes)

# >>> NOVO: SALVA A TABELA BINÁRIA PARA VOCÊ ACESSAR <<<
nome_binaria = 'tabela_binaria_completa.xlsx'
df_binaria.to_excel(nome_binaria, index=False)
print(f"SUCESSO! Tabela binária salva em '{nome_binaria}'.")

# --- 3. MACHINE LEARNING ---
target = 'Densidade de Corrente (mA/cm²)'
colunas_para_ignorar = ['Formula_MOF', 'Condicoes_Sintese', 'Durabilidade', 'Tendência de Desempenho']
X = df_binaria.drop(columns=[target] + colunas_para_ignorar, errors='ignore')
X = X.select_dtypes(include=[np.number])
y = df_binaria[target]

print("Treinando Inteligência Artificial...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# --- 4. EXPORTAR RESULTADOS ---
df['Predição_IA'] = rf.predict(X)
nome_saida = 'resultado_completo_20_linhas.xlsx'
df.to_excel(nome_saida, index=False)

print(f"SUCESSO! Resultado final salvo em '{nome_saida}'.")
print("="*50)

# --- RANKING DE IMPORTÂNCIA ---
importancias = pd.DataFrame({
    'Característica': X.columns,
    'Importância': rf.feature_importances_
})
importancias = importancias.sort_values(by='Importância', ascending=False)
importancias['Importância (%)'] = (importancias['Importância'] * 100).round(2).astype(str) + '%'

print("\nRANKING DE IMPORTÂNCIA ( EM % ):")
print("-" * 50)
print(importancias[['Característica', 'Importância (%)']].head(5).to_string(index=False))
