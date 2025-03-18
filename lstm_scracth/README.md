#  LSTM from Scratch

This project implements a Long Short-Term Memory (LSTM) neural network from scratch using PyTorch. The model is designed to predict stock prices based on historical data.

## Features
- Custom LSTM implementation using PyTorch `nn.Module`
- Training using Mean Squared Error (MSE) loss and Adam optimizer
- Basic dataset preparation and DataLoader for training
- Predictions on stock data for different companies

## Implementation Details

### LSTM from Scratch
The LSTM model is implemented from scratch using PyTorch `nn.Parameter` to define trainable weights and biases for:
- Forget Gate
- Input Gate
- Candidate Memory
- Output Gate

Each step processes the input sequence to update short-term and long-term memory states.

### Model Architecture
```python
class LSTM_Scratch(nn.Module):
    def __init__(self):
        super().__init__()
        mean, std = 0.0, 1.0

        # Forget Gate
        self.wa1 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.wa2 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.ba1 = nn.Parameter(torch.tensor(0.0))

        # Input Gate
        self.wb1 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.wb2 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.bb1 = nn.Parameter(torch.tensor(0.0))

        # Candidate Memory
        self.wc1 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.wc2 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.bc1 = nn.Parameter(torch.tensor(0.0))

        # Output Gate
        self.wd1 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.wd2 = nn.Parameter(torch.normal(mean=torch.tensor(mean), std=torch.tensor(std)))
        self.bd1 = nn.Parameter(torch.tensor(0.0))

    def lstm_unit(self, input_value, long_memory, short_memory):
        long_memory_percent = torch.sigmoid((short_memory * self.wa1) + (input_value * self.wa2) + self.ba1)
        potential_memory_percent = torch.sigmoid((short_memory * self.wb1) + (input_value * self.wb2) + self.bb1)
        potential_long_memory = torch.tanh((short_memory * self.wc1) + (input_value * self.wc2) + self.bc1)
        updated_long_memory = (long_memory * long_memory_percent) + (potential_memory_percent * potential_long_memory)
        output_short_memory = torch.sigmoid((short_memory * self.wd1) + (input_value * self.wd2) + self.bd1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_short_memory
        return updated_long_memory, updated_short_memory

    def forward(self, input):
        batch_size, seq_len = input.shape
        long_memory = torch.zeros(batch_size, device=input.device)
        short_memory = torch.zeros(batch_size, device=input.device)

        for t in range(seq_len):
            long_memory, short_memory = self.lstm_unit(input[:, t], long_memory, short_memory)

        return short_memory
```

### Training the Model
The model is trained using:
- Mean Squared Error (MSE) Loss
- Adam optimizer
- 500 epochs with a learning rate of 0.01

```python
def train_model(model, dataloader, num_epochs=500, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    print("Training complete!")
```

### Sample Predictions
After training, the model is used to predict stock price trends:
```python
print("\nPredictions:")
print("Company A pred_value:", model(torch.tensor([[0.0, 0.5, 0.25, 1.0]])).detach().item())
print("Company B pred_value:", model(torch.tensor([[1.0, 0.5, 0.25, 1.0]])).detach().item())
```

### Training Output Example
```
Epoch [1/500], Loss: 0.8386
Epoch [101/500], Loss: 0.0357
Epoch [201/500], Loss: 0.0054
Epoch [301/500], Loss: 0.0026
Epoch [401/500], Loss: 0.0016
Training complete!

Predictions:
Company A pred_value: 0.9819
Company B pred_value: 0.9841
```

## Requirements
- Python 3.7+
- PyTorch

