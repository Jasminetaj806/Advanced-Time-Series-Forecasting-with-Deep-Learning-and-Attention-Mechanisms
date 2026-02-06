def evaluate(model, loader):
    model.eval()
    preds = []
    attns = []

    with torch.no_grad():
        for x, y in loader:
            out, attention = model(x)
            preds.append(out)
            attns.append(attention)

    return preds, attns