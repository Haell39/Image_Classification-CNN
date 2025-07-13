# Classificação de Imagens CIFAR-10 com CNN

## Visão Geral
Este projeto implementa uma **Rede Neural Convolucional (CNN)** para classificar imagens do **conjunto de dados CIFAR-10**, que consiste em 60.000 imagens coloridas de 32x32 pixels, abrangendo 10 classes (por exemplo, avião, carro, pássaro). O modelo é construído usando **PyTorch** e realiza a classificação de imagens através de treinamento e avaliação no conjunto de dados.

## Funcionalidades
* **Conjunto de Dados**: CIFAR-10 com 50.000 imagens para treinamento e 10.000 para teste.
* **Modelo**: CNN com duas camadas convolucionais, max-pooling e camadas totalmente conectadas.
* **Treinamento**: 30 épocas com otimizador SGD (taxa de aprendizado: 0.001, momentum: 0.9).
* **Avaliação**: Atinge aproximadamente 65.49% de acurácia no conjunto de teste.
* **Inferência**: Suporta a classificação de imagens personalizadas com pré-processamento.

## Requisitos
* Python 3.x
* PyTorch
* Torchvision
* PIL (Pillow)
* NumPy

## Uso

### Configuração:
Instale as dependências: `pip install torch torchvision pillow numpy`
O conjunto de dados CIFAR-10 será baixado automaticamente pelo script.

### Treinamento:
Execute o script para treinar o modelo por 30 épocas.
Os pesos do modelo são salvos em `trained_net.pth`.

### Teste:
Avalie o modelo no conjunto de teste para calcular a acurácia.
Use imagens personalizadas para inferência colocando-as em `test_images/` e atualizando `image_paths` no script.

### Inferência:
Pré-processe as imagens para 32x32 com normalização.
Carregue o modelo treinado e preveja as classes para novas imagens.

## Exemplo
```python
image_paths = ['test_images/img.png', 'test_images/img2.png']
for img in image_paths:
    image = load_image(img)
    output = net(image)
    predicted_class = class_name[torch.max(output, 1)[1].item()]
    print(f'Classe Prevista: {predicted_class}')
