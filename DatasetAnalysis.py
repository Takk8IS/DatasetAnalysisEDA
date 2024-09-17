import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List, Dict
from dataclasses import dataclass
import re
import unidecode

# Configurações de estilo do Matplotlib
plt.style.use('default')  # Usa o estilo padrão do Matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

@dataclass(frozen=True)
class ModelMetrics:
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    feature_importance: np.ndarray

def load_data(file_path: str) -> pd.DataFrame:
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_extension.lower() == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado. Por favor, use arquivos .csv ou .xlsx.")

def correct_word(word: str, corrections: Dict[str, str]) -> str:
    word = unidecode.unidecode(word.lower())
    return corrections.get(word, word)

def correct_text(text: str, corrections: Dict[str, str]) -> str:
    words = re.findall(r'\w+', text.lower())
    return ' '.join(correct_word(word, corrections) for word in words).capitalize()

def grammar_correction(df: pd.DataFrame) -> pd.DataFrame:
    corrections = {
        'curso': 'curso',
        'sexo': 'sexo',
        'setor': 'setor',
        'serie': 'serie',
        'horasorientadas': 'horas_orientadas',
        'diassuspensao': 'dias_suspensao',
        'perdeu_residencia': 'perdeu_residencia',
        'perdeuresidencia': 'perdeu_residencia',
        'matricula': 'matricula'
    }

    df.columns = [correct_text(col, corrections) for col in df.columns]

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: correct_text(str(x), corrections) if pd.notnull(x) else x)

    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = grammar_correction(df)
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    target_column = next((col for col in df.columns if 'perdeu' in col.lower() and 'residencia' in col.lower()), None)
    if not target_column:
        raise ValueError("Coluna alvo 'perdeu_residencia' não encontrada no DataFrame")

    X = df.drop(['matricula', target_column], axis=1, errors='ignore')
    y = df[target_column]

    return X, y

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"Pontuações F1 da validação cruzada: {cv_scores}")
    print(f"Média da pontuação F1 da validação cruzada: {np.mean(cv_scores):.4f}")

    rf_classifier.fit(X_train_scaled, y_train)
    y_pred = rf_classifier.predict(X_test_scaled)

    perm_importance = permutation_importance(rf_classifier, X_test_scaled, y_test, n_repeats=10, random_state=42)

    return ModelMetrics(
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        feature_importance=perm_importance.importances_mean
    )

def plot_histogram(data: pd.Series, title: str, xlabel: str, ylabel: str):
    plt.figure()
    sns.histplot(data, kde=True, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_metrics(metrics: List[float], metric_names: List[str]):
    plt.figure()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    sns.barplot(x=metric_names, y=metrics, palette=colors)
    plt.title('Métricas de Desempenho', pad=20)
    plt.ylabel('Pontuação')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Mapa de Calor de Correlação', pad=20)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm: np.ndarray):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão', pad=20)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance: np.ndarray, feature_names: List[str]):
    sorted_idx = feature_importance.argsort()
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, len(feature_names) * 0.5))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.title('Importância das Features', pad=20)
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()

def analyze_dataset(file_path: str):
    df = load_data(file_path)
    X, y = preprocess_data(df)

    print("Análise Exploratória dos Dados:")
    print(f"Número de amostras: {len(df)}")
    print(f"Número de features: {len(X.columns)}")
    print(f"\nDistribuição da variável alvo 'perdeu_residencia':\n{y.value_counts(normalize=True)}")

    print("\nEstatísticas descritivas das features numéricas:")
    print(X.describe())

    metrics = train_and_evaluate_model(X, y)

    print("\nResultados da Classificação:")
    print(f"Precisão: {metrics.precision:.4f}")
    print(f"Revocação: {metrics.recall:.4f}")
    print(f"Pontuação F1: {metrics.f1:.4f}")

    plot_metrics([metrics.precision, metrics.recall, metrics.f1], ['Precisão', 'Revocação', 'Pontuação F1'])

    print("\nVisualizações:")
    for column in X.columns:
        plot_histogram(X[column], f'Distribuição de {column}', column, 'Frequência')

    plot_correlation_heatmap(X)
    plot_confusion_matrix(metrics.confusion_matrix)
    plot_feature_importance(metrics.feature_importance, X.columns.tolist())

    print("\nAnálise de Correlação:")
    horas_orientadas_col = next((col for col in X.columns if 'horas' in col.lower() and 'orientadas' in col.lower()), None)
    if horas_orientadas_col:
        print(X.corr()[horas_orientadas_col].sort_values(ascending=False))
    else:
        print("Coluna 'horas_orientadas' não encontrada")

    print("\nInterpretação dos Resultados:")
    print("1. As features mais importantes para prever a perda de residência são:")
    for feature, importance in sorted(zip(X.columns, metrics.feature_importance), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   - {feature}: {importance:.4f}")

    print("\n2. Observações sobre as correlações:")
    if horas_orientadas_col:
        high_corr = X.corr()[horas_orientadas_col].sort_values(ascending=False)
        print(f"   - A feature mais correlacionada com '{horas_orientadas_col}' é '{high_corr.index[1]}' com correlação de {high_corr.values[1]:.2f}")
    else:
        print("   - Não foi possível analisar correlações com 'horas_orientadas'")

    print("\n3. Distribuição das features:")
    for column in X.columns:
        skew = X[column].skew()
        print(f"   - {column}: Assimetria = {skew:.2f}")
        if abs(skew) > 1:
            print(f"     Atenção: '{column}' apresenta assimetria significativa.")

    print("\nRecomendações:")
    print("1. Considere realizar engenharia de features para melhorar o modelo.")
    print("2. Investigue mais a fundo as features com alta importância e correlação.")
    print("3. Para features com alta assimetria, considere transformações (ex: log) ou técnicas de normalização.")
    print("4. Revise o processo de coleta de dados para minimizar erros ortográficos e inconsistências.")
    print("5. Verifique se todas as colunas esperadas estão presentes no dataset e se seus nomes estão corretos.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python dataset_analysis.py <caminho_para_o_dataset>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_dataset(file_path)
