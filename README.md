-----

# README: Classificação de Imagens CIFAR-10 com CNN

-----

## Visão Geral

Este projeto implementa uma **Rede Neural Convolucional (CNN)** usando **PyTorch** para classificar imagens do conjunto de dados **CIFAR-10**. O CIFAR-10 contém 60.000 imagens coloridas de 32x32 pixels, divididas em 10 classes (ex: avião, carro, pássaro). O objetivo é treinar um modelo para identificar essas classes.

## Funcionalidades

  * **Dataset**: CIFAR-10 (50.000 imagens de treino, 10.000 de teste) com pré-processamento de normalização.
  * **Modelo**: CNN com duas camadas convolucionais, max-pooling e três camadas totalmente conectadas.
  * **Treinamento**: 30 épocas com otimizador **SGD** (learning rate: 0.001, momentum: 0.9) e função de perda CrossEntropyLoss.
  * **Avaliação**: Acurácia calculada no conjunto de teste.
  * **Inferência**: Classificação de imagens customizadas após pré-processamento.

## Requisitos

  * Python 3.x
  * PyTorch
  * Torchvision
  * PIL (Pillow)
  * NumPy

Instale as dependências com: `pip install torch torchvision pillow numpy`

## Uso

### 1\. Preparação

Os dados do CIFAR-10 serão baixados automaticamente na primeira execução para a pasta `./data`.

### 2\. Treinamento

O script treina o modelo por 30 épocas, exibindo a perda por época. O modelo treinado é salvo em `trained_net.pth`.

```python
# Exemplo de loop de treinamento
for epoch in range(30):
    # ... código de treinamento ...
    print(f'Loss: {running_loss / len(train_loader):.4f}')

torch.save(net.state_dict(), 'trained_net.pth')
```

### 3\. Carregar o Modelo

Para carregar o modelo salvo:

```python
net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth'))
```

### 4\. Avaliação

A acurácia do modelo no conjunto de teste é calculada:

```python
# ... código de avaliação ...
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
```

### 5\. Inferência

Para classificar suas próprias imagens (`test_images/img.png`, `test_images/img2.png`):

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# ... new_transform e load_image ...

image_paths = ['test_images/img.png', 'test_images/img2.png']
images = [load_image(img) for img in image_paths]

net.eval()
with torch.no_grad():
    for image in images:
        output = net(image)
        _, predicted = torch.max(output, 1)
        print(f'Classe Prevista: {class_name[predicted.item()]}')
```

## Resultados

  * **Perda Final de Treinamento**: Aproximadamente `0.4550`
  * **Acurácia no Teste**: **68.48%**
  * **Previsões de Exemplo**:
      * `img.png`: dog
      * `img2.png`: plane
