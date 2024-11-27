 <H1>Como funciona Apredizado por Reforço?</H1>
 <a>De forma simples, o ator (quem faz a ação) escolhe, a partir de uma política (cérebro), uma ação com base no estado do ambiente. Essa ação neste estado resulta em uma recompensa, boa ou ruim, e atualiza a política. Abordaremos superficialmente os cálculos disso nos demais tópicos.</a>

 <H1>Objetivos:</H1>
 <a>O objetivo deste trabalho é realizar uma análise do modelo de aproximação desenvolvido, avaliando seu desempenho em ambientes de aprendizado por reforço com características desafiadoras, como o BipedalWalker-v3 e CartPole-v1. Para isso, serão analisados fatores como o tempo de convergência, estabilidade do aprendizado e a recompensa média obtida pelos agentes treinados. Essa comparação busca identificar as forças e limitações de cada abordagem, contribuindo para a escolha de métodos mais adequados a diferentes aplicações práticas.</a>
 <H1>Materiais e Métodos:</H1>
 <a>Nesta seção, são descritos os recursos, ferramentas e procedimentos utilizados para realizar avaliação do código desenvolvido no ambiente de aprendizado por reforço BipedalWalker-v3.</a>
  
 <h3>Ambiente de Simulação</h3>

 <a>O ambiente utilizado foram aqueles fornecidos pela biblioteca Gymnasium. Este ambiente simula um robô bípede que deve aprender a caminhar de forma eficiente, exigindo a coordenação de múltiplos motores. Cada estado do ambiente é representado por um vetor de 24 dimensões, que inclui informações como posição, velocidade e ângulos das juntas. As ações do agente consistem em valores contínuos que controlam a força aplicada em quatro motores, tornando o problema desafiador devido à sua natureza de controle contínuo e alta dimensionalidade.</a>

 <h3>Métricas de Avaliação</h3>

<a>
 O algoritmo foi avaliado com base nas seguintes métricas:

     Recompensa acumulada por episódio: Mede o desempenho do agente ao longo dos episódios.
     Taxa de convergência: Número de episódios necessários para atingir um desempenho estável.
     Estabilidade: Avaliação da variância na recompensa ao longo do treinamento.

 Os experimentos foram conduzidos utilizando um computador, garantindo maior eficiência no treinamento. Os resultados obtidos foram registrados e analisados para identificar as diferenças de desempenho entre os dois métodos.</a>

 <H1>Discussão sobre o assunto:</H1>
 <a>
 Geralmente ouvimos falar de Deep SARSA e DQN na comunidade de IA, isso porque são tipos de RL mais difundidos, no entanto suas áreas de atuação são parecidas. A unica diferença entre DSARSA e DQN são o tipo de política:

     *On-Poliicy:
     O método que o DSARSA usa. Na maioria das vezes é sensível à política inicial. Bom em ambientes de muita variância (estocásticos) e por isso a escolha deste método.
     *Off-Policy:
     O método que o DQN usa. Pouco sensível à política inicial, o que não leva a mínimos locais. Bom em ambientes determinísticos e por isso a comparação.
 Sobre o modelo desenvolvido: Apesar de ser uma junção de q-tables e redes neurais, o código não usa necessariamente nenhuma dessas aproximações. Na verdade, o modelo gera pequenas variações aleatórias nos parâmetros (theta) do agente. Essas variações são chamadas de deltas, e para cada delta, o agente executa um episódio de interação com o ambiente. A recompensa de cada variação (positiva ou negativa) é registrada, e as variações que produzem as maiores recompensas são selecionadas para atualizar os parâmetros do modelo. Portanto, o modelo é baseado numa tabela de pesos Theta e não há redes neurais e nem tabelas Q, se demonstrando eficiente com pouco uso de processamento.
  Ainda mais, existem parâmetros que, apesar da parecerem atrasar o processo de aprendizado, adiantam e muito. Nos dois métodos acima foram escritos linhas de código de ruído nos movimentos do agente. Esses ruídos ajudam na exploração do ambiente quando o que agente faria é 'exploitar'. Na prática, um exemplo muito simples é quando o agente se mantém parado, e, assim, não recebendo recompensa ou recebendo menos que um erro fatal. Esse tipo de decisão geralmente está relacionado com os mínimos locais, mas essa discução se direciona às derivadas parciais e matrizes, portanto não abordarei aqui.
 </a>
 <H1>Resultados</H1>
 <h2>Quanto a Convergência</h2>
 <a>
  Foi testado o algorítmo durante 1500 episódios consecutivos. Durante esses episódios foram resgatados dados de recompensa total e o próprio número do episódio em relação à recompensa acumulada. Com esses dados foi possível encenar um gráfico com a biblioteca matplotlib da seguinte forma:
 </a>
 <img src="/images/recompensa_acumulada.png">
 <a>Todo esse processo resultou em um desvio padrão de aproximadamente 3.39, no entanto, após o episódio 263, o desvio padrão cai para aproximadamente 1 e após o 500 o desvio padrão fica abaixo de 0.61, expressando um resultado ótimo</a>
 <h2>Quanto a implementação</h2>
 <a>
  Nesta parte abordaremos a matemática relacionada ao algorítmo de aprendizado discutido.
  Em primeiro lugar, precisamos entender como ele funciona na prática:
  Inicia-se 3 matrizes 4x24 com valores 0: Step, Delta e Theta. Delta é uma matriz com valores aleatórios que impactam na direção dos valores de Theta durante a execução (aumento ou diminuição do valor escalar). O episódio, após finalizado, retorna as recompensas acumuladas de cada direção (R_neg e R_pos). Mas, então, como sabemos se Theta deve descer ou subir? Simples, fazemos a fórmula para atualizar os parâmetros theta que pode ser escrita da seguinte forma:
 </a>

$`\theta = \theta _{antigo} + \frac{lr}{(\Delta_{\text{max}} \times \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i - \mu)^2})} \times [S_{antigo} + \delta R \times \Delta]`$

O que exatamente isso significa? Simples:

$`\theta`$ é a matriz de pesos

$`\theta _{antigo}`$ é a matriz teta que não foi modificada pela equação a seguir, a antiga matriz $`\theta`$

$`lr`$ é a taxa de aprendizado

$`\Delta`$ é a matriz de variações nos pesos $`theta`$

$`\sum_{i=1}^{N} (r_i - \mu)^2`$ é o desvio padrão das recompensas

$`S_{antigo}`$ é a antiga matriz Step que continham os valores de Step modificados com a diferença entre as recompensas de $`\Delta`$ positivo e negativo. Em suma, é ele quem decide se $`\theta`$ deve subir ou abaixar

$`\delta R`$ é a diferença entre as recompensas citadas acima

$`\Delta`$ é a propria matriz Delta que contém os valores adicionados ou retirados de theta

$`\Delta _{max}`$ é a quantidade de Deltas que serão selecionados pela recompensa, os "melhores Deltas"

<a>Portanto, o algorítmo não passa de uma operação linear que, por reforço, "acha" os mínimos da função sem necessidade de programar _Bacpropagation_ e nem _Postpropagation_, que aumentaria em dificuldade e uso de processamento, mas, em contrapartida, é menos eficiente que uma rede neural completa. Pela proximidade com redes neurais e métodos como CMA, DSARSA e DQN eu vou chamá-la, por economia de palavras, de Neural Aproximation, ou apenas NA.</a>
 
 <h2>Quanto ao uso de processamento e memória</h2>
 <a>
 </a>
 <h1>Próximos Passos e o Futuro do Laboratório de Ideação (LABI)</h1>
 <a>
 </a>
