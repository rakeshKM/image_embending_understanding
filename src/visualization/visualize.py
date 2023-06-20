from visualization.confusion import confusion
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def visualize_results(Results,outfolder):
    fig = confusion(Results.y_true,Results.y_pred,Results.classes)
    fig.savefig(outfolder + 'confusion.png', dpi=300)