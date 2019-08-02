# Lookahead Optimizer
The implement of [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).
## PyTorch
Usage:
```python
optimizer = torch.opti.SGD(model.parameters(), lr=0.01) # Any optimizer
optimizer = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead

# train
optimizer.zero_grad()
loss_fn(model(data), target).backward()
optimizer.step()
```
