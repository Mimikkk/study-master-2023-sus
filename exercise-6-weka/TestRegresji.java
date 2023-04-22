import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

class TestRegresji {
  public static void printStats(String dataset, Classifier classifier, Evaluation evaluation) {
    try {
      var result = new StringBuffer();
      result.append("Dataset	 : " + dataset + "\n");
      result.append("Classifier: " + Utils.toCommandLine(classifier) + "\n\n");
      result.append("Dataset	 : " + dataset + "\n");
      result.append("classifier.toString():\n" + classifier.toString() + "\n");
      result.append("evaluation.toSummaryString():\n" + evaluation.toSummaryString() + "\n");
      System.out.println(result.toString());
    } catch (Exception e) { e.printStackTrace(); }
  }


  public static void main(String[] args) throws Exception {
    for (var i = 1; i <= 50; ++i) {
      var name = "M5P";
      var dataset = "./data/regression.arff";
      var options = new Vector<String>();
      options.add("-R");
      options.add("-M");
      options.add(Integer.toString(i));

      var classifier = AbstractClassifier.forName(name, options.toArray(new String[options.size()]));

      var instances = new Instances(new BufferedReader(new FileReader(dataset)));

      instances.setClassIndex(instances.numAttributes() - 1);
      classifier.buildClassifier(instances);

      var evaluation = new Evaluation(instances);
      evaluation.crossValidateModel(classifier, instances, 10, instances.getRandomNumberGenerator(1));

      printStats(dataset, classifier, evaluation);
    }
  }
};
