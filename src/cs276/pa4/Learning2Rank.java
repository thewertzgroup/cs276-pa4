package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Learning2Rank {

	
	public static Classifier train(String train_data_file, String train_rel_file, int task, Map<String,Double> idfs) throws Exception {
	    System.err.println("## Training with feature_file =" + train_data_file + ", rel_file = " + train_rel_file + " ... \n");
	    Classifier model = null;
	    Learner learner = null;
    
 		if (task == 1) {
			learner = new PointwiseLearner();
		} else if (task == 2) {
		  boolean isLinearKernel = false;
			learner = new PairwiseLearner(isLinearKernel);
		} else if (task == 3) {
			
			/* 
			 * @TODO: Your code here, add more features 
			 * */
			boolean isLinearKernel = false;
			learner = new PairwiseLearner(isLinearKernel, true, false, true);
			System.err.println("Task 3");
			
		} else if (task == 4) {
			
			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");
			
		}
 		
		/* Step (1): construct your feature matrix here */
		Instances data = learner.extract_train_features(train_data_file, train_rel_file, idfs);
		
		/* Step (2): implement your learning algorithm here */
		model = learner.training(data);
	 		
	  return model;
	}

	 public static Map<String, List<String>> test(String test_data_file, Classifier model, int task, Map<String,Double> idfs) throws Exception{
		 	System.err.println("## Testing with feature_file=" + test_data_file + " ... \n");
		    Map<String, List<String>> ranked_queries = new HashMap<String, List<String>>();
		    Learner learner = null;
	 		if (task == 1) {
				learner = new PointwiseLearner();
			} else if (task == 2) {
			  boolean isLinearKernel = true;
				learner = new PairwiseLearner(isLinearKernel);
			} else if (task == 3) {

				/* 
				 * @TODO: Your code here, add more features 
				 * */
				boolean isLinearKernel = false;
				learner = new PairwiseLearner(isLinearKernel, true, false, true);
				System.err.println("Task 3");
				
			} else if (task == 4) {
				
				/* 
				 * @TODO: Your code here, extra credit 
				 * */
				System.err.println("Extra credit");
				
			}
		 
	 		/* Step (1): construct your test feature matrix here */
	 		TestFeatures tf = learner.extract_test_features(test_data_file, idfs);
	 		
	 		/* Step (2): implement your prediction and ranking code here */
			ranked_queries = learner.testing(tf, model);
			
		    return ranked_queries;
	 }
	
	

	/* This function output the ranking results in expected format */
	public static void writeRankedResultsToFile(Map<String,List<String>> ranked_queries, PrintStream ps) {
	    for (String query : ranked_queries.keySet()){
	      ps.println("query: " + query.toString());

	      for (String url : ranked_queries.get(query)) {
	        ps.println("  url: " + url);
	      }
	    }
	}
	

	public static void main(String[] args) throws Exception {
	    if (args.length != 4 && args.length != 5) {
	      System.err.println("Input arguments: " + Arrays.toString(args));
	      System.err.println("Usage: <train_data_file> <train_rel_file> <test_data_file> <task> [ranked_out_file]");
	      System.err.println("  ranked_out_file (optional): output results are written into the specified file. "
	          + "If not, output to stdout.");
	      return;
	    }

	    String train_data_file = args[0];
	    String train_rel_file = args[1];
	    String test_data_file = args[2];
	    int task = Integer.parseInt(args[3]);
	    String ranked_out_file = "";
	    if (args.length == 5){
	      ranked_out_file = args[4];
	    }
	    
	    /* Populate idfs */
	    String dfFile = "df.txt";
	    Map<String,Double> idfs = null;
	    try {
	      idfs = Util.loadDFs(dfFile);
	    } catch(IOException e){
	      e.printStackTrace();
	    }
	    // process idfs to be ready: currently it is just document frequencies
	    int totalDocCount = 98998; 
		// add the special term to account for terms not in training corpus 
	    if(!idfs.containsKey(Util.specialTerm)) // the term is new
		{			 							
			idfs.put(Util.specialTerm, 0.0); 
		//	System.out.println("success");
		}
		// Make idf using df and do laplace smoothing 
		for (String term : idfs.keySet()) {
			/*
			 * @//TODO : Your code here
			 */
			double idf = Math.log((double)(totalDocCount+1)/(idfs.get(term)+1.0));// add 1 smoothing: laplace 		 		
			idfs.put(term, idf); // overwrite the document frequency with the smoothened idf
			}
		
		
	    
	    
	    /* Train & test */
	    System.err.println("### Running task" + task + "...");		
	    Classifier model = train(train_data_file, train_rel_file, task, idfs);

	    /* performance on the training data */
	    Map<String, List<String>> trained_ranked_queries = test(train_data_file, model, task, idfs);
	    String trainOutFile="tmp.train.ranked";
	    writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));
	    NdcgMain ndcg = new NdcgMain(train_rel_file);
	    System.err.println("# Trained NDCG=" + ndcg.score(trainOutFile));
	    (new File(trainOutFile)).delete();
      
	    Map<String, List<String>> ranked_queries = test(test_data_file, model, task, idfs);
	 
	    /* Output results */
	    
	    if(ranked_out_file.equals("")){ /* output to stdout */
	      writeRankedResultsToFile(ranked_queries, System.out);
	    } else { 						/* output to file */
	      try {
	        writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
	        String test_rel_file = ".\\data\\pa4.rel.dev"; 
	        NdcgMain test_ndcg = new NdcgMain(test_rel_file);
		    System.err.println("# Test NDCG=" + test_ndcg.score(ranked_out_file));
	      } catch (FileNotFoundException e) {
	        e.printStackTrace();
	      }
	    }
	}
}
