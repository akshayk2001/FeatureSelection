import math


def nearestNeighborAlgorithm(numInstances, oneOutInstance, features, myInstances):
   nearestNeighbor = -1 #Nearest neighbor initially set to -1
   nearestNeighborDistance = float("inf") #distance of nearest neighbor initially set to infinity
   for i in range(numInstances):
       if(i == oneOutInstance): #If the one out instance is the ith iteration then don't do anything this iteration
           pass
       else:
           myDistance = 0
           for k in range(len(features)):
               myDistance += pow((myInstances[i][features[k]] - myInstances[oneOutInstance][features[k]]), 2) #Calculcate distance given distance = √(sum((x-y)^2)) where x and y are two values
           myDistance = math.sqrt(myDistance) #Take the square root of the distance
           if(myDistance < nearestNeighborDistance): #If the newly calculated distance is lower than the nearest neighbor distance then its the new nearest neighbor
               nearestNeighborDistance = myDistance #Set the nearest neighbor distance to this distance
               nearestNeighbor = i #Nearest neighbor is this iteration
   return nearestNeighbor #We return our nearest neighbor




def leaveOneCrossValidation(numInstances, features, myInstances, feature):
   if(feature > 0): #If we are performing forward selection
       featuresList = list(features) #Make a list version of the set because sets are not subscriptable
       featuresList.append(feature) #Add the feature to the list
   elif(feature < 0): #If we are performing backward elimination
       feature = feature * -1 #Change the feature back to positive
       features.remove(feature) #Remove the feature from the set
       featuresList = list(features)
       features.add(feature) #Add the feature back to the set since we don't want to change the original set that was passed in
   else: #If we are not appending or removing an item
       featuresList = list(features)
   numCorrectlyClassified = 0 #Used to keep track for number of instances correctly classified
   for i in range(numInstances):
       oneOutInstance = i #For each fold we take an instance
       nearestNeighbor = nearestNeighborAlgorithm(numInstances, oneOutInstance, featuresList, myInstances) #Calculate the nearest neighbor to this one out instance
       if(myInstances[nearestNeighbor][0] == myInstances[oneOutInstance][0]): #If the nearest neighbor class label matches the one out class label
           numCorrectlyClassified += 1 #Increment the number of correctly classified labels
   accuracy = (numCorrectlyClassified / numInstances) #Accuracy is the number of correctly classified labels divided by the number of total instances
   accuracy *= 100 #Multiply by 100 to get percentage number
   return round(accuracy, 1) #Round accuracy to the tenths place
  

