import tensorflow_addons as tfa

def triplet_loss(margin = 1.0):
    def inner_triplet_loss_objective(y_true, y_pred):
        labels = y_true
        embeddings = y_pred
        return tfa.losses.triplet_semihard_loss(y_true=labels, y_pred=embeddings,margin=margin)
    return inner_triplet_loss_objective