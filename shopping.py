import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

    evidence, labels = load_data("shopping.csv")
    #print(f"First row of evidence: {evidence[0]}")

    #This function estimate impact of each feature on the model
    #feature_importance(model, X_test, y_test)
    #PageValues is the most important feature with around 0.07 score
    #This is because when a purchase is made, the pagevalue as a value
    #So if pagevalue is not null, it is more likely that a purchase is made
    #Even if it's not sure that the purchase is made, it is a good indicator


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """    
    evidence = []
    labels = []
    count = 0
    count_positive = 0
    # Mapping months to numerical values
    month_map = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Append evidence as a list
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_map[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ])
            
            # Append label
            labels.append(1 if row["Revenue"] == "TRUE" else 0)
            if row["Revenue"] == "TRUE":
                count_positive += 1
            count += 1
    # Apply scaling to the evidence (features)        
    scaler = StandardScaler()
    evidence_scaled = scaler.fit_transform(evidence) 

    print (f"total: {count}, positive: {count_positive}")

    return evidence_scaled, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Initialize the k-NN model with the specified number of neighbors
    model = KNeighborsClassifier(n_neighbors=1)#, metric='manhattan'
  
    # Fit the model on the evidence and labels
    model.fit(evidence, labels)
    return model

#I tried using K-fold but result are similars
def train_model_k_fold(evidence, labels):   
    # Initialize k-fold cross-validator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Initialize the model
    model = KNeighborsClassifier(n_neighbors=1, weights='distance')
    
    for train_index, test_index in kf.split(evidence):
        # Split the data into training and testing sets for this fold
        train_evidence = [evidence[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        # Train the model on the training data of the current fold
        model.fit(train_evidence, train_labels)
    
    # Return the model trained on all folds
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialize counters
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Iterate through labels and predictions
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            if predicted == 1:
                true_positive += 1
            else:
                false_negative += 1
        elif actual == 0:
            if predicted == 0:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate sensitivity (true positive rate)
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Calculate specificity (true negative rate)
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    print(accuracy)
    return sensitivity, specificity

def feature_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance = result.importances_mean
    for i, v in enumerate(importance):
        print(f"Feature: {i}, Score: {v:.5f}")

if __name__ == "__main__":
    main()
    




