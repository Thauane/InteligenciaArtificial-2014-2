# Codigo original de autoria de Krishnamurthy Koduvayur Viswanathan
# Modificado por Hendrik Macedo
 
from __future__ import division
import collections
import math
 
class Model: 
        def __init__(self, arffFile):
                self.trainingFile = arffFile
                self.featureNameList = []       #this is to maintain the order of features as in the arff
                self.featureCounts = collections.defaultdict(lambda: 1)#contains tuples of the form (label, feature_name, feature_value)
                self.featureVectors = []        #contains all the values and the label as the last entry
                self.labelCounts = collections.defaultdict(lambda: 0)  
                                 
        def GetValues(self):
                file = open(self.trainingFile, 'r')
                for line in file:
                        if line[0] != '@':  #start of actual data
                                self.featureVectors.append(line.strip().lower().split(','))
                        else:   #feature definitions
                                self.featureNameList.append(line.strip().split()[1])
                file.close()
 
        def TrainClassifier(self):
                for fv in self.featureVectors:
                        self.labelCounts[fv[len(fv)-1]] += 1 #udpate count of the label
                        for counter in range(0, len(fv)-1):
                                self.featureCounts[(fv[len(fv)-1], self.featureNameList[counter], fv[counter])] += 1
 
        def TestClassifier(self, arffFile):
                file = open(arffFile, 'r')
                for line in file:
                    vector = line.strip().lower().split(',')
                    print "classificado como *" + self.Classify(vector) + "* dado que o correto seria *" + vector[len(vector) - 1] + "*\n" 
 
        def Classify(self, featureVector):      #featureVector is a simple list like the ones that we use to train
                probabilityPerLabel = {}
                for label in self.labelCounts:
                        #logProb = 0
                        prob = 1;
                        for featureValue in featureVector:
                                #logProb += math.log(self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]/self.labelCounts[label])
                                prob *= self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]/self.labelCounts[label]
                        #probabilityPerLabel[label] = round(math.exp(logProb),5)
                        probabilityPerLabel[label] = prob;
                print probabilityPerLabel
                return max(probabilityPerLabel, key = lambda classLabel: probabilityPerLabel[classLabel])
                              
        
if __name__ == "__main__":
        model = Model("zoo.txt")
        model.GetValues()
        model.TrainClassifier()
        model.TestClassifier("zoo_teste.txt")
