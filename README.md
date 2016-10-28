# Large-Scale-Machine-Learning-with-Spark-for-Marketing
This project aims to address prediction problems for marketing, including customer churn prediction, using Spark and Scala.

Customer Churn Dataset:

The customer churn dataset is provided by the UC Irvine machine-learning repository hosted by SGI. The dataset contains 5,000 subscribers. The full set of fields are given as below:

1) state

2) account length

3) area code

4) phone number

5) international plan

6) voice mail plan

7) number vmail messages

8) total day minutes

9) total day calls

10) total day charge

11) total eve minutes

12) total eve calls

13) total eve charge

14) total night minutes

15) total night calls

16) total night charge

17) total intl minutes

18) total intl calls

19) total intl charge

20) number customer service calls

21) churned (class label)

# Package the Scala project with SBT

1. Download sbt.rar and unpack it into C:\project\

2. In the folder: C:\project\sbt\, run: $ sbt assembly

3. Copy the executable JAR file from the folder C:\project\sbt\target\scala-2.11\ to the folder C:\project\, copy the data file credit_data.txt to the folder C:\project\

4. Run the Spark program: $ spark-submit CustomerChurnPrediction-assembly-1.0.jar

