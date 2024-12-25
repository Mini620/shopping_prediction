# shopping_prediction
Classification model which predict if someone is likely to buy on a web site

I have the 3 main function working : load_data, train_model and evaluate
but I also added to 2 more : feature_importance which estimate impact of each feature on the model and train_model_k_fold to train with k-fold

After using feature_importance I can say pageValues is the most important feature with around 0.07 score
This is because when a purchase is made, the pagevalue as a value
So if pagevalue is not null, it is more likely that a purchase is made
Even if it's not sure that the purchase is made, it is a good indicator

My model provide around 45% True positive rate(sensitivity), 90-95% True negatives(specificity) and 0.85 accuracy 
When I change the N neighbors value true positive decrease and True negative tend toward 100%, which break the balance in the results.
With 12330 lines and only 1908 with a sale, I estimated the chance of randomly guessing a sale at 15% which mean my model is at worst 3 times better
I think those results seems satisfying because we are trying to estimate human behavior.
I'm glad my model isn't just guessing no every time like in a previous model where i was unable to fix it.
