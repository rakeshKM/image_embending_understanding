from sklearn.metrics import accuracy_score, confusion_matrix

class Results:
    def __init__(self,y_true,y_pred,classes) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.metrics = {}
        self.get_metrics()
    
    def get_metrics(self):
        self.accuracy()
        self.confusion()

    def print_metrics(self):
        for key in self.metrics:
            print(f"{key} = {self.metrics[key]}")
    
    def accuracy(self):
        self.metrics["accuracy"] = accuracy_score(self.y_true,self.y_pred)
        
    def confusion(self):
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_true,self.y_pred)