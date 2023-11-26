// Define a função sigmoid
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
// Define a função sigmoid derivada
function sigmoidDerivative(x) {
    return x * (1 - x);
}

// Define a classe Neuron
class Neuron {
    constructor() {
        this.weights = [];
        this.bias = Math.random();
        this.output = 0;
        this.error = 0;
    }

    // Inicializa os pesos
    initWeights(numWeights) {
        for (let i = 0; i < numWeights; i++) {
            this.weights.push(Math.random());
        }
    }

    // Calcula a saída do neurônio
    calculateOutput(inputs) {
        let sum = this.bias;
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        this.output = sigmoid(sum);
    }
}

// Define a classe NeuralNetwork
class NeuralNetwork {
    constructor(numInputs, numHidden, numOutputs) {
        this.inputLayer = [];
        this.hiddenLayer = [];
        this.outputLayer = [];

        // Inicializa a camada de entrada
        for (let i = 0; i < numInputs; i++) {
            this.inputLayer.push(new Neuron());
        }

        // Inicializa a camada oculta
        for (let i = 0; i < numHidden; i++) {
            let neuron = new Neuron();
            neuron.initWeights(numInputs);
            this.hiddenLayer.push(neuron);
        }

        // Inicializa a camada de saída
        for (let i = 0; i < numOutputs; i++) {
            let neuron = new Neuron();
            neuron.initWeights(numHidden);
            this.outputLayer.push(neuron);
        }
    }

    // Calcula a saída da rede neural
    calculateOutput(inputs) {
        let hiddenOutputs = [];
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            this.hiddenLayer[i].calculateOutput(inputs);
            hiddenOutputs.push(this.hiddenLayer[i].output);
        }

        let output = [];
        for (let i = 0; i < this.outputLayer.length; i++) {
            this.outputLayer[i].calculateOutput(hiddenOutputs);
            output.push(this.outputLayer[i].output);
        }

        return output;
    }

    // Define a função de treinamento
    train(inputs, targets, learningRate) {
        // Calcula a saída da rede neural
        let outputs = this.calculateOutput(inputs);
    
        // Calcula o erro da camada de saída
        for (let i = 0; i < this.outputLayer.length; i++) {
            let neuron = this.outputLayer[i];
            neuron.error = (targets[i] - neuron.output) * sigmoidDerivative(neuron.output);
        }
    
        // Calcula o erro da camada oculta
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            let neuron = this.hiddenLayer[i];
            let errorSum = 0;
            for (let j = 0; j < this.outputLayer.length; j++) {
                errorSum += this.outputLayer[j].error * this.outputLayer[j].weights[i];
            }
            neuron.error = errorSum * sigmoidDerivative(neuron.output);
        }
    
        // Atualiza os pesos e os bias da camada de saída
        for (let i = 0; i < this.outputLayer.length; i++) {
            let neuron = this.outputLayer[i];
            for (let j = 0; j < neuron.weights.length; j++) {
                neuron.weights[j] += neuron.error * this.hiddenLayer[j].output * learningRate;
            }
            neuron.bias += neuron.error * learningRate;
        }
    
        // Atualiza os pesos e os bias da camada oculta
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            let neuron = this.hiddenLayer[i];
            for (let j = 0; j < neuron.weights.length; j++) {
                neuron.weights[j] += neuron.error * inputs[j] * learningRate;
            }
            neuron.bias += neuron.error * learningRate;
        }
    }
}

// Cria a rede neural
let nn = new NeuralNetwork(2, 3, 1);

// Treina a rede neural
for (let i = 0; i < 200000; i++) {
    nn.train([0, 0], [0], 0.1);
    nn.train([0, 1], [1], 0.1);
    nn.train([1, 0], [1], 0.1);
    nn.train([1, 1], [0], 0.1);
}

// Testa a rede neural
console.log(nn.calculateOutput([0, 0])); // Deve imprimir algo próximo de 0
console.log(nn.calculateOutput([0, 1])); // Deve imprimir algo próximo de 1
console.log(nn.calculateOutput([1, 0])); // Deve imprimir algo próximo de 1
console.log(nn.calculateOutput([1, 1])); // Deve imprimir algo próximo de 0
