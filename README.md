 <H1>Como funciona Apredizado por Reforço?</H1>
 <a>De forma simples, o ator (quem faz a ação) escolhe, a partir de uma política (cérebro), uma ação com base no estado do ambiente. Essa ação neste estádo resulta em uma recompensa, boa ou ruim, e atualiza a política. Abordaremos superficialmente os cálculos disso nos demais tópicos.</a>

 <H1>Objetivos:</H1>
 <a>O objetivo deste trabalho é realizar uma análise comparativa entre os métodos Deep SARSA e Deep Q-learning, avaliando seu desempenho em ambientes de aprendizado por reforço com características desafiadoras, como o BipedalWalker-v3 e CartPole-v1. Para isso, serão analisados fatores como o tempo de convergência, estabilidade do aprendizado e a recompensa média obtida pelos agentes treinados. Essa comparação busca identificar as forças e limitações de cada abordagem, contribuindo para a escolha de métodos mais adequados a diferentes aplicações práticas.</a>
 <H1>Materiais e Métodos:</H1>
 <a>Nesta seção, são descritos os recursos, ferramentas e procedimentos utilizados para realizar a comparação entre os métodos Deep SARSA e Deep Q-learning no ambiente de aprendizado por reforço BipedalWalker-v3.
 Ambiente de Simulação

 O ambiente utilizado foi o BipedalWalker-v3, fornecido pela biblioteca Gymnasium. Este ambiente simula um robô bípede que deve aprender a caminhar de forma eficiente, exigindo a coordenação de múltiplos motores. Cada estado do ambiente é representado por um vetor de 24 dimensões, que inclui informações como posição, velocidade e ângulos das juntas. As ações do agente consistem em valores contínuos que controlam a força aplicada em quatro motores, tornando o problema desafiador devido à sua natureza de controle contínuo e alta dimensionalidade.
 Algoritmos Avaliados

 Foram implementados dois algoritmos:

     *Deep Q-learning (DQL):
         -O agente utiliza uma rede neural para aproximar a função de valor Q(s,a)Q(s,a), que estima a recompensa esperada para cada par estado-ação.
         -O treinamento segue a abordagem off-policy, utilizando a máxima recompensa futura esperada, mesmo que a ação escolhida não seja a executada pelo agente.

     *Deep SARSA:
         -Diferente do DQL, este método atualiza a função de valor Q(s,a)Q(s,a) com base na sequência de estados e ações efetivamente executados pelo agente, adotando uma abordagem on-policy.

 Ambos os algoritmos foram implementados utilizando o PyTorch para construção das redes neurais e o Gymnasium para a simulação do ambiente.
 Estrutura das Redes Neurais

 As redes neurais para ambos os algoritmos possuem a seguinte arquitetura:

     Camada de entrada: 24 neurônios, correspondentes às dimensões do estado.
     Duas camadas ocultas:
         Primeira camada: 128 neurônios, com função de ativação ReLU.
         Segunda camada: 64 neurônios, também com função de ativação ReLU.
     Camada de saída: 4 neurônios, representando as ações contínuas no ambiente.

 Estratégia de Treinamento

 O treinamento foi realizado utilizando um Replay Buffer com capacidade de 10.000 amostras, permitindo que o agente aprenda a partir de experiências passadas de maneira mais eficiente. Em cada iteração:

     O agente interage com o ambiente e armazena a transição (s,a,r,s′,d)(s,a,r,s′,d) no buffer.
     Um lote de transições é amostrado aleatoriamente para atualizar os pesos da rede neural.
     A função de perda utilizada foi o erro quadrático médio (MSE) entre a estimativa Q(s,a)Q(s,a) e a recompensa-alvo calculada.

 Hiperparâmetros Utilizados

     Tamanho do Replay Buffer: 10.000.
     Tamanho do lote (batch size): 64.
     Taxa de aprendizado (learning rate): 0,001.
     Taxa de exploração (epsilon):
         Inicial: 1.0.
         Decaimento: 0,995 por episódio.
         Mínimo: 0,01.
     Fator de desconto (γγ): 0,99.

 Métricas de Avaliação

 Os algoritmos foram avaliados com base nas seguintes métricas:

     Recompensa acumulada por episódio: Mede o desempenho do agente ao longo dos episódios.
     Taxa de convergência: Número de episódios necessários para atingir um desempenho estável.
     Estabilidade: Avaliação da variância na recompensa ao longo do treinamento.

 Os experimentos foram conduzidos utilizando um computador, garantindo maior eficiência no treinamento das redes neurais. Os resultados obtidos foram registrados e analisados para identificar as diferenças de desempenho entre os dois métodos.</a>

 <H1>Discussão sobre o assunto:</H1>
 <a>
 Geralmente não ouvimos falar de Deep SARSA na comunidade de IA, isso porque o tipo de rede DQN, anterior ao DSARSA, foi mais difundido, no entanto suas áreas de atuação são parecidas. A unica diferença entre DSARSA e DQN são o tipo de política:

     *On-Poliicy:
     O método que o DSARSA usa. Na maioria das vezes é sensível à política inicial. Bom em ambientes de muita variância (estocásticos) e por isso a escolha deste método.
     *Off-Policy:
     O método que o DQN usa. Pouco sensível à política inicial, o que não leva a mínimos locais. Bom em ambientes determinísticos e por isso a comparação.
 Ainda mais, existem parâmetros que, apesar da parecerem atrasar o processo de aprendizado, adiantam e muito. Nos dois métodos foram escritos linhas de código de ruido nos movimentos do agente. Esses ruidos ajudam na exploração do ambiente quando o que agente faria é 'exploitar'. Na prática, um exemplo muito simples é quando o agente se mantém parado, e, assim, não recebendo recompensa ou recebendo menos que um erro fatal. Esse tipo de decisão geralmente está relacionado com os mínimos locais, mas essa discução se direciona às derivadas parciais e matrizes, portanto não abordarei aqui.
 </a>
 <H1>Resultados</H1>
 <h2>Quanto a Convergência</h2>
 <a>

 </a>
 <h2>Quanto a implementação</h2>
 <a>

 </a>
 <h2>Quanto ao uso de processamento e memória</h2>
 <a>

 </a>
 <h1>Próximos Passos e o Futuro do Laboratório de Ideação (LABI)</h1>
 <a>
     
 </a>
