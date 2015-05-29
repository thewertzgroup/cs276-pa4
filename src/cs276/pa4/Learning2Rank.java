package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs276.pa4.Learner.Features;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class Learning2Rank
{
	private static List<Features> features = null;

	public static Classifier train(String train_data_file, String train_rel_file, int task, Map<String,Double> idfs) 
	{
		System.err.println("## Training with feature_file = " + train_data_file + ", rel_file = " + train_rel_file + " ... \n");
		Classifier model = null;
		Learner learner = null;

		if (task == 1) 
		{
			learner = new PointwiseLearner();
		} 
		else if (task == 2) 
		{
			boolean isLinearKernel = false;
			learner = new PairwiseLearner(isLinearKernel);
		} 
		else if (task == 3) 
		{

			/* 
			 * @TODO: Your code here, add more features 
			 * */
			System.err.println("Task 3");

			boolean isLinearKernel = false;
			learner = new PairwiseLearner(isLinearKernel);
			learner.setFeatures(features);
		} 
		else if (task == 4) 
		{

			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");

		}
		
		Learner.setIDFs(idfs);

		/* Step (1): construct your feature matrix here */
		Instances data = learner.extract_train_features(train_data_file, train_rel_file);

		/* Step (2): implement your learning algorithm here */
		model = learner.training(data);

		return model;
	}

	
	public static class PairwiseClassifier
	{
		public double C_exp;
		public double gamma_exp;
		public Classifier model;
		
		PairwiseClassifier(double C_exp, double gamma_exp, Classifier model)
		{
			this.C_exp = C_exp;
			this.gamma_exp = gamma_exp;
			this.model = model;
		}
	}

	
	public static List<PairwiseClassifier> getPairwiseClassifiers(String train_data_file, String train_rel_file, int task, Map<String,Double> idfs) 
	{
		List<PairwiseClassifier> pairwiseClassifiers = new ArrayList<>();
		
		for (int C_exp = -3; C_exp <= 3; C_exp++)
		{
			for (int gamma_exp = -7; gamma_exp <= -1; gamma_exp++)
			{
				System.err.println("## Training with feature_file = " + train_data_file + ", rel_file = " + train_rel_file + " ... \n");
				System.err.println("\nnon-linear SVM: C = 2^" + C_exp + " gamma = 2^" + gamma_exp + "\n\n");
				
				Classifier model = null;
				Learner learner = null;

				boolean isLinearKernel = false;
				learner = new PairwiseLearner(Math.pow(2.0,  C_exp), Math.pow(2.0, gamma_exp), isLinearKernel);
				if (task == 3) learner.setFeatures(features);
				Learner.setIDFs(idfs);

				/* Step (1): construct your feature matrix here */
				Instances data = learner.extract_train_features(train_data_file, train_rel_file);

				/* Step (2): implement your learning algorithm here */
				model = learner.training(data);

				pairwiseClassifiers.add(new PairwiseClassifier(C_exp, gamma_exp, model));
			}
		}
		return pairwiseClassifiers;
	}

	/*
	 * 
	 */
	public static Map<String, List<String>> test(String test_data_file, Classifier model, int task, Map<String,Double> idfs)
	{
		System.err.println("## Testing with feature_file = " + test_data_file + " ... \n");
		Map<String, List<String>> rankedQueries = new HashMap<String, List<String>>();
		Learner learner = null;
		if (task == 1) 
		{
			learner = new PointwiseLearner();
		} 
		else if (task == 2) 
		{
			boolean isLinearKernel = true;
			learner = new PairwiseLearner(isLinearKernel);
		} 
		else if (task == 3) 
		{

			/* 
			 * @TODO: Your code here, add more features 
			 * */
			System.err.println("Task 3");

			boolean isLinearKernel = false;
			learner = new PairwiseLearner(isLinearKernel);
			learner.setFeatures(features);
		}
		else if (task == 4) 
		{

			/* 
			 * @TODO: Your code here, extra credit 
			 * */
			System.err.println("Extra credit");

		}
		
		Learner.setIDFs(idfs);

		/* Step (1): construct your test feature matrix here */
		TestFeatures tf = learner.extract_test_features(test_data_file);

		/* Step (2): implement your prediction and ranking code here */
		rankedQueries = learner.testing(tf, model);

		return rankedQueries;
	}


	/* This function output the ranking results in expected format */
	public static void writeRankedResultsToFile(Map<String,List<String>> ranked_queries, PrintStream ps) 
	{
		for (String query : ranked_queries.keySet()){
			ps.println("query: " + query.toString());

			for (String url : ranked_queries.get(query)) {
				ps.println("  url: " + url);
			}
		}
	}


	public static void main(String[] args) throws IOException 
	{
		if (args.length != 4 && args.length != 5 && args.length != 6) 
		{
			System.err.println("Input arguments: " + Arrays.toString(args));
			System.err.println("Usage: <train_data_file> <train_rel_file> <test_data_file> <task> [ranked_out_file]");
			System.err.println("  ranked_out_file (optional): output results are written into the specified file. "
					+ "If not, output to stdout.");
			return;
		}

		boolean gridSearch = false;
		String train_data_file = args[0];
		String train_rel_file = args[1];
		String test_data_file = args[2];
		int task = Integer.parseInt(args[3]);
		String ranked_out_file = "";
		String test_rel_file = "";
		if (args.length >= 5)
		{
			ranked_out_file = args[4];
		}
		if (args.length == 6)
		{
			gridSearch = true;
			test_rel_file = args[5];
		}

		/* Populate idfs */
		String dfFile = "df.txt";
		Map<String,Double> idfs = null;
		try 
		{
			idfs = Util.getIDFs( Util.loadDFs(dfFile) );
		} 
		catch(IOException e)
		{
			e.printStackTrace();
		}
		
		List<List<Features>> featureLists = new ArrayList<>();
		featureLists.add(new ArrayList<Features>());
		featureLists.add(Arrays.asList(Features.BM25));
		featureLists.add(Arrays.asList(Features.SmallWindow));
		featureLists.add(Arrays.asList(Features.PageRank));
		featureLists.add(Arrays.asList(Features.BM25, Features.SmallWindow));
		featureLists.add(Arrays.asList(Features.BM25, Features.PageRank));
		featureLists.add(Arrays.asList(Features.SmallWindow, Features.PageRank));
		featureLists.add(Arrays.asList(Features.BM25, Features.SmallWindow, Features.PageRank));
		
		for (List<Features> curFeatures : featureLists)
		{
			//features =  Arrays.asList(Features.BM25, Features.SmallWindow, Features.PageRank);
			features = curFeatures;
			System.err.println("\n\n\n################ Current Features ################\n");
			System.err.println("################ : " + features + "\n\n");
	
			/* Train & test */
			System.err.println("### Running task " + task + " ...");		
			
			Classifier model = null;
			if (gridSearch)
			{
				System.err.println("### Running grid search ...");		
	
				List<PairwiseClassifier> classifiers = getPairwiseClassifiers(train_data_file, train_rel_file, task, idfs);
				
				for (PairwiseClassifier pairwiseClassifer : classifiers)
				{
					System.err.println("\n\nTesting non-linear SVM model with C = 2^" + pairwiseClassifer.C_exp + " gamma = 2^" + pairwiseClassifer.gamma_exp + "\n\n");
					model = pairwiseClassifer.model;
					
	//				Learner.reset(); // Reset cached parameters.
					testModel(train_data_file, train_rel_file, test_data_file, ranked_out_file, model, task, idfs);
					
					NdcgMain ndcgTest = new NdcgMain(test_rel_file);
					System.err.println("# Test NDCG=" + ndcgTest.score(ranked_out_file));
					(new File(ranked_out_file)).delete();
				}
			}
			else
			{
				System.err.println("### Running model train/test ...");		
	
				model = train(train_data_file, train_rel_file, task, idfs);
	
				testModel(train_data_file, train_rel_file, test_data_file, ranked_out_file, model, task, idfs);	
			}
		}

	}
	
	
	private static void testModel(String train_data_file, String train_rel_file, String test_data_file, String ranked_out_file, Classifier model, int task, Map<String,Double> idfs) throws IOException
	{
		/* performance on the training data */
		Map<String, List<String>> trained_ranked_queries = test(train_data_file, model, task, idfs);
		
		String trainOutFile="tmp.train.ranked";
		writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));
		
		NdcgMain ndcg = new NdcgMain(train_rel_file);
		System.err.println("# Trained NDCG=" + ndcg.score(trainOutFile));
		(new File(trainOutFile)).delete();

//		Learner.reset(); // Reset cached parameters.
		Map<String, List<String>> ranked_queries = test(test_data_file, model, task, idfs);

		/* Output results */
		if (ranked_out_file.equals(""))
		{ 
			/* output to stdout */
			writeRankedResultsToFile(ranked_queries, System.out);
		} 
		else 
		{ 	
			/* output to file */
			try 
			{
				writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
			} 
			catch (FileNotFoundException e) 
			{
				e.printStackTrace();
			}
		}
	}

}
