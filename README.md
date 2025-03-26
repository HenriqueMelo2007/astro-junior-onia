# Missão AstroJúnior: explorando novos mundos!🪐
# Desafio feito pela **1ª Olímpiada Nacional de Inteligência Artificial ( ONIA )**. 
![Space Banner](/assets/banner.webp)

No ano de 2075, a humanidade expandiu suas viagens pelo universo, e jovens cadetes da Academia Espacial foram selecionados para a grande Missão AstroJúnior. Seu objetivo: explorar planetas desconhecidos e coletar dados para ajudar cientistas da Terra a entender esses novos mundos.

Cada equipe recebeu um conjunto de sensores de alta tecnologia para medir diferentes aspectos dos planetas visitados. Os dados coletados foram processados e codificados em uma escala numérica própria de modo a refletir as seguintes grandezas:

1. TempMédia - Temperatura média do planeta (em graus Celsius);
1. Gravidade - Intensidade da gravidade em relação à Terra;
1. PressãoAtm - Pressão atmosférica (em atmosferas terrestres);
1. Radiação - Nível de radiação presente no planeta;
1. ComposiçãoAr - Percentual de gases essenciais para a vida humana;
1. Hidratação - Disponibilidade de água líquida;
1. Vegetação - Presença de organismos vegetais;
1. Fauna - Diversidade de formas de vida animal;
1. SoloFértil - Capacidade do solo de sustentar plantação;
1. Ventos - Intensidade dos ventos planetários;
1. Luas - Número de luas orbitando o planeta;
1. Magnetismo - Força do campo magnético do planeta;
1. ClimaEstável - Estabilidade das condições climáticas ao longo do tempo;

Com base nesses dados, os cientistas classificarão os planetas em cinco categorias principais:

- **Classe 0:** Planeta Deserto - Muito quente ou frio, sem água e com poucas chances de vida.
- **Classe 1:** Planeta Vulcânico - Alta atividade geotérmica e atmosfera hostil.
- **Classe 2:** Planeta Oceânico - Coberto por vastos oceanos, com possibilidades de vida aquática.
- **Classe 3:** Planeta Florestal - Rico em vegetação, similar à Terra em muitos aspectos.
- **Classe 4:** Planeta Gelado - Extremamente frio, mas com possibilidade de vida subterrânea ou em oceanos sob o gelo.

Agora, cadete, sua missão é analisar os dados coletados e ajudar a classificar cada planeta corretamente. O futuro da exploração espacial está em suas mãos!

> A planilha de treinamento (treino.csv) fornecida tem 10.501 linhas e 13 colunas.

Após treinar o seu modelo, você deve realizar a predição da planilha de teste (teste.csv). A planilha de teste fornecida tem 4.501 linhas e 13 características das instâncias (colunas). Ela contém informações semelhantes à planilha de treinamento, mas é fornecida sem os rótulos (“target”), ou seja, sem as categorias de planetas.

Quando você julgar que criou um modelo competitivo envie suas predições para as categorias dos planetas pela plataforma utilizando um arquivo com a extensão .csv no formato exato como descrito a seguir:

- Deve conter precisamente 4.501 linhas e 2 colunas
- Na primeira célula da primeira coluna deve estar escrito a palavra "id"
- Na primeira célula da segunda coluna deve estar escrito a palavra "target"
Cada linha desse arquivo, com exceção da primeira, deve conter na célula da primeira coluna o “id” (ou seja, um número correspondente à instância) e na célula da segunda coluna o “target” (0, 1, 2, 3, 4)

Os resultados serão avaliados pelo desempenho de suas predições sobre o conjunto de teste, utilizando a métrica “Medida-F”.

A Medida-F é a média harmônica entre as métricas de Precisão e Revocação. Em outras palavras, a Medida-F é uma métrica que avalia o desempenho de um modelo preditivo de modo a trazer um número único que indique a sua qualidade geral.

# Solução

## 📁[**Arquivo de código**](main.py)

1. Carregamento dos dados
1. Análise exploratória
1. Pré-processamento
1. Construção dos pipelines de modelagem
1. Otimização dos hiperparâmetros
1. Seleção e treinamento do melhor modelo
1. Geração das previsões

## 📁[**Arquivo Jupyter**](jupyther.ipynb)

Exibição de gráficos para maior compreensão dos dados
