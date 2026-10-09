def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def arr_sum(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def arr_prod(a, b):
    return [a * bi for bi in b]

def bias(x):
    return [row + [1] for row in x]

class Neuron():
    def __init__(self, dim):
        self.dim = dim
        self.weights = [0] * (dim+1)
    
    def fit(self, data, labels):
        data = bias(data)
        wrong = True
        count = 0
        while wrong:
            wrong = False
            for x, y in zip(data, labels):
                if y * dot(x, self.weights) <= 0:
                    self.weights = arr_sum(self.weights, arr_prod(y, x))
                    wrong = True
            count += 1
        print(count)
    def predict(self, x):
        x = bias(x)
        results = []
        for i in x:
            results.append(-1 if dot(i, self.weights) <= 0 else 1)
        return results

X = [
    [ 0.2,  0.3],
    [ 0.5, -0.4],
    [-0.3,  0.7],
    [ 0.1, -0.6],
    [-0.4, -0.2],
    [ 1.5,  1.2],
    [ 2.0,  0.8],
    [ 1.1,  1.8],
    [ 0.9,  1.1],
    [ 1.8,  1.6],
]
y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

model = Neuron(2)
model.fit(X, y)

X_test = [
    [ 0.3, -0.1],
    [-0.2, -0.5],
    [ 0.0,  0.5],
    [-0.5,  0.2],
    [ 0.4, -0.7],
    [ 1.3,  1.0],
    [ 1.7,  1.5],
    [ 1.0,  1.4],
    [ 2.1,  1.1],
    [ 1.4,  1.9],
]
y_test = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

preds = model.predict(X_test)

score = 0
for i in range(len(preds)):
    if preds[i] == y_test[i]:
        score += 1
print(f"Accuracy: {score/len(y_test)}")