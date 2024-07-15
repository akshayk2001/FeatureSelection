import math
import copy
import validationAndNeighbor as nearestNeighbor


numInstances = 0  #Number of instances, global because used in multiple functions
numFeatures = 0    #Number of features, global because used in multiple functions


def normalizeInstances(myInstances):
   global numFeatures #Because this global variable is used in this function
   meanList = []   #Includes the mean of each column as an entry in the list
   stDevList = []  #Includes the std of each column as an entry in the list
   for i in range(1, numFeatures + 1): #Starting from 1 to exclude class column
       meanSum = 0 #Used to collect sum of each column
       for row in myInstances: #For each line
           meanSum += row[i] #Sum all the values in the column
       meanList.append(meanSum / numInstances) #Divide by number of instances to get mean for that particular column and add to mean list


   for i in range(1, numFeatures + 1):
       stDevSum = 0
       for row in myInstances:
           stDevSum += pow((row[i] - meanList[i - 1]), 2) #Subtract value from mean and square result for each row
       stDevList.append(math.sqrt(stDevSum / numInstances)) #Take square root of sum divided by number of instances to get std for that column and add to stDev list


   for i in range(0, numInstances):
       for j in range(1, numFeatures + 1):
           myInstances[i][j] = ((myInstances[i][j] - meanList[j-1]) / stDevList[j-1]) #Change each feature value to normalized feature value
   return myInstances #return newly normalized instances


def forwardSelection(normInstances):
   global numFeatures
   currentFeatures = set() #new set for current features
   finalFeatures = set() #new set for final features
   highestAccuracy = 0.0 #Used to note the highest accuracy
   myAccuracy = nearestNeighbor.leaveOneCrossValidation(numInstances, currentFeatures, normInstances, 0) #Calculate the accuracy of the feature set using leave one out cross validation
   print("Using no features gives an accuracy of", myAccuracy, "%") #Print accuracy for empty set
   for i in range(numFeatures): #for each feature
       addFeature = -1 #new feature to be added initially set to -1 for default
       print("On level %d of the search tree" % (i+ 1))
       localAddFeature = -1 #Local feature to be added initially set to -1 for default
       localAccuracy = 0.0 #Highest local accuracy initially set to 0.0
       for j in range(1, numFeatures + 1): #j will take on the value of each feature
           if(j not in currentFeatures): #meaning we have not yet selected this feature
               whatIfSet = copy.copy(currentFeatures) #shallow copy the current feature set in order to use for printing purposes
               whatIfSet.add(j) #Add the new potential feature
               myAccuracy = nearestNeighbor.leaveOneCrossValidation(numInstances, currentFeatures, normInstances, j)
               if(len(whatIfSet) == 0): #If we have an empty set
                   print("Using no features gives an accuracy of", myAccuracy, "%") #Nicer syntax when printing for empty set
               else:
                   print("Using feature(s)", whatIfSet, "gives an accuracy of", myAccuracy, "%") #Prints accuracy in the case of adding the feature
               if(myAccuracy > highestAccuracy): #if we have a new all time high accuracy
                   highestAccuracy = myAccuracy #Set highest to the accuracy we just calculcated since it is higher
                   addFeature = j #feature to be added is j
               elif(myAccuracy > localAccuracy): #if we have a new local maxima
                   localAccuracy = myAccuracy #Set highest local to the accuracy we just calculcated since it is higher than current local accuracy
                   localAddFeature = j #local feature to be added is j
       if(addFeature >= 0): #if we have a new feature to be added that is going to bring the all time accuracy higher
           currentFeatures.add(addFeature) #Add the feature into from the current feature set
           finalFeatures.add(addFeature) #Add the feature into the final feature set
           print("Feature(s): ", currentFeatures, "were best, accuracy is", highestAccuracy, "%")
       else: #All time high accuracy was not surpassed in this round but keep searching in case this is a local maxima
           print("Warning, Accuracy has decreased! Continuing search in the event of local maxima")
          currentFeatures.add(localAddFeature) #Add the feature into the current feature set
           print("Feature(s): ", currentFeatures, "were best, accuracy is", localAccuracy, "%") #Feature subset that gives highest local accuracy
   print("Search completed.")
   print("Best feature subset is", finalFeatures, ", which has an accuracy of", highestAccuracy, "%") #Feature subset that gives highest accuracy


def backwardElimination(normInstances):
   global numFeatures
   currentFeatures = set()
   finalFeatures = set()
   highestAccuracy = 0.0
   myAccuracy = 0.0
   for i in range(numFeatures):
       currentFeatures.add(i+1) #Starting with all the features
       finalFeatures.add(i+1) #Starting with all the features
   myAccuracy = nearestNeighbor.leaveOneCrossValidation(numInstances, currentFeatures, normInstances, 0)
   print("Using feature(s)", currentFeatures, "gives an accuracy of", myAccuracy, "%") #Prints accuracy for the full set of features before we remove anything
   for i in range(numFeatures):
       removeFeature = -1 #new feature to be removed initially set to -1 for default
       print("On level %d of the search tree" % (i+ 1))
       localRemoveFeature = -1
       localAccuracy = 0.0
       for j in range(1, numFeatures + 1):
           if(j in currentFeatures): #If we still have not removed this feature
               whatIfSet = copy.copy(currentFeatures)
               whatIfSet.remove(j) #remove feature in question for printing purpose
               myAccuracy = nearestNeighbor.leaveOneCrossValidation(numInstances, currentFeatures, normInstances, (-1*j))
               if(len(whatIfSet) == 0):
                   print("Using no features gives an accuracy of", myAccuracy, "%")
               else:
                   print("Using feature(s)", whatIfSet, "gives an accuracy of", myAccuracy, "%") #Prints the accuracy in the case of removing the feature
               if(myAccuracy > highestAccuracy):
                   highestAccuracy = myAccuracy
                   removeFeature = j #feature to be removed is j
               elif(myAccuracy > localAccuracy):
                   localAccuracy = myAccuracy
                   localRemoveFeature = j #local feature to be removed is j
       if(removeFeature >= 0): #if we have a new feature to be removed that is going to bring all time accuracy higher
           currentFeatures.remove(removeFeature) #Remove the feature from the current feature set
           finalFeatures.remove(removeFeature) #Remove the feature from the final feature set
           print("Feature(s): ", currentFeatures, "were best, accuracy is", highestAccuracy, "%")
       else:
           currentFeatures.remove(localRemoveFeature) #Remove the feature from the current feature set only since we still have a different better feature set somewhere else
           print("Warning, Accuracy has decreased! Continuing search in the event of local maxima")
           print("Feature(s): ", currentFeatures, "were best, accuracy is", localAccuracy, "%")
   print("Search completed.")
   print("Best feature subset is", finalFeatures, ", which has an accuracy of", highestAccuracy, "%")




def main():
   global numInstances
   global numFeatures
   myInstances = [] #Will be the whole collection of values in file
   print("Hello and Welcome to the Feature Selection Program\n")
   fileName = input("Please enter the test case file name: ") #Grab input from user for file name
   searchType = "" #Variable used for gathering user input on algorithm type
   while (searchType != "F" and searchType != "B"): #While the user entered a value algorithm type
       searchType = input("""Please enter the name of the search algorithm you wish to use (F for Forward Selection, B for Backward Elimination): """) #Grab input from user
   try:
       myFile = open(fileName, 'r') #Try to open the file
   except:
        raise IOError('The given file does not exist.') #If file is not found then raise an IO error
   firstInstance = myFile.readline() #Read first line of file
   numFeatures = len(firstInstance.split()) - 1 #Split line into elements by space and subtract 1 because of class column
   myFile.seek(0) #Return cursor to start position of file to avoid calculating wrong number of instances
   fileContent = myFile.read() #Read all contents of file as string
   fileList = fileContent.split("\n") #Split up file content string by newline
   for row in fileList: #Each row represents a line in the file
       if(len(row) > 0): #If we have a row of actual values
           numInstances = numInstances + 1 #For each row we have a new instance
   for i in range(numInstances):
       myInstances.append([]) #Initializing an empty array of size num of instances
   incrementor = -1 #Used to assign values to each row; -1 because gets incremented before being used
   for line in fileList: #For each line in the file
       incrementor+=1 #Increment for each line to move down each row in our myInstances array
       valueList = line.split("  ") #Split up the row string by 2 character spaces to get just the value
       for value in valueList: #For each value in the row
           if(len(value) > 0): #If we have an actual number value and not a space
               myInstances[incrementor].append(float(value)) #Adding values to each row
   normInstances = normalizeInstances(myInstances) #Normalize instances and return new array
   print("This dataset has %d features with %d instances." % (numFeatures, numInstances))
   if (searchType == "F"): #If we have selected forward selection
       forwardSelection(normInstances)
   else: #We selected backward elimination
       backwardElimination(normInstances) 
   myFile.close() #Close the file since we are done with it


if __name__ == '__main__': #Prevents main from being called when the code is imported as a module
   main()
