package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner 
{
	private LibSVM model;
	
	public PairwiseLearner(boolean isLinearKernel)
	{
		try
		{
			model = new LibSVM();
		} 
		catch (Exception e)
		{
			e.printStackTrace();
		}

		if (isLinearKernel)
		{
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}

	public PairwiseLearner(double C, double gamma, boolean isLinearKernel)
	{
		try
		{
			model = new LibSVM();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		model.setCost(C);
		model.setGamma(gamma); // only matter for RBF kernel
		if(isLinearKernel)
		{
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}
	
	
	private TestFeatures extract_dataset(String data_file)
	{
		return extract_dataset(data_file, null);
	}


	private TestFeatures extract_dataset(String data_file, String relevance_file)
	{
		TestFeatures testFeatures = new TestFeatures();
		testFeatures.index_map = new HashMap<>();
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));

		String[] labels = {"-1", "1"};
		attributes.add(new Attribute("class", Arrays.asList(labels)));
		testFeatures.features = new Instances(relevance_file != null ? "train_dataset" : "test_dataset", attributes, 0);
		
		/* Set last attribute as target */
		testFeatures.features.setClassIndex(testFeatures.features.numAttributes() - 1);
		
		TestFeatures standardizedFeatures = standardize(PointwiseLearner.extract_dataset(data_file, relevance_file));
		
		/* Add data */
		Map<Query,List<Document>> queryMap = null;
		Map<String, Map<String, Double>> relMap = null;
		try 
		{
			queryMap = Util.loadTrainData(data_file);
			relMap 	 = relevance_file != null ? Util.loadRelData(relevance_file) : null;
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			throw new RuntimeException("Unable to load signal data, or relevance data.", e);
		}

		// Set scorer for base class.
		if (null == scorer)
		{
			scorer = new BM25Scorer(idfs, queryMap);
		}
		
		// Iterate over queries / document pairs and compute five-dimensional training vectors.
		int    positive = 0;
		int    negative = 0;
		int         rel = 0;
		double     iRel = 0.0;
		double     jRel = 0.0;
		for (Query query : queryMap.keySet())
		{
			// Maps the URL pair to the row index of the test matrix to retrieve for prediction.
			Map<String, Integer> urlIndexMap = new HashMap<>(); 

			List<Document> docs = queryMap.get(query);
			for (int i=0; i<docs.size(); i++)
			{
				for (int j=i+1; j<docs.size(); j++)
				{
					if (relMap != null)
					{
						iRel = relMap.get(query.query).get(docs.get(i).url);
						jRel = relMap.get(query.query).get(docs.get(j).url);
					}
					
					// Do not make pairwise ranking facts out of either pairs 
					// of documents with the same relevance score.
					if (relMap != null && iRel == jRel) continue;
					
					double[] iVector = toVector(standardizedFeatures.features.get(standardizedFeatures.index_map.get(query.query).get(docs.get(i).url)));
					double[] jVector = toVector(standardizedFeatures.features.get(standardizedFeatures.index_map.get(query.query).get(docs.get(j).url)));
					
					double[] vector = subtractArrays(iVector, jVector);

					String indexUrl = docs.get(i).url + " " + docs.get(j).url;
					if (relMap != null)
					{
						rel = (int)Math.signum(iRel - jRel);
						if ((rel < 0 && positive < negative) || (rel > 0 && positive > negative))
						{
							rel *= -1;
							negateArray(vector);
							indexUrl = docs.get(j).url + " " + docs.get(i).url;
						}
						
						if (rel > 0) positive++;
						if (rel < 0) negative++;
						if (rel == 0) throw new RuntimeException("'0' relevance when lable should be '-1' or '1'.");
					}

					Instance instance = new DenseInstance(6);					
				    instance.setDataset(testFeatures.features);
				    setInstanceValues(instance, vector);
				    
				    if (relMap != null)
				    {
						String label = Integer.toString(rel);
					    instance.setValue(5, label); 
				    }
				    
				    testFeatures.features.add(instance);
					
					urlIndexMap.put(indexUrl, testFeatures.features.size()-1);
				}
			}
			
			testFeatures.index_map.put(query.query,  urlIndexMap);
		}
		
		return testFeatures;
	}


	private TestFeatures standardize(TestFeatures testFeatures) 
	{
		//System.err.println("\nDataset:\n\n" + testFeatures.features);
		
		TestFeatures standardized = new TestFeatures();
		
		standardized.index_map = testFeatures.index_map;
		
		Standardize filter = new Standardize();	  
		try {
			filter.setInputFormat(testFeatures.features);
			standardized.features = Filter.useFilter(testFeatures.features, filter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//System.err.println("\nStandardized Dataset:\n\n" + standardized.features);

		return standardized;
	}

	private void setInstanceValues(Instance instance, double[] vector) 
	{
	    for(int i = 0; i < vector.length; i++)
	    {
	    	instance.setValue(i, vector[i]);	
	    }
	}

	
	private double[] toVector(Instance instance)
	{
		return instance.toDoubleArray();
	}
	
	
	private double[] subtractArrays(double[] x, double[] y) 
	{
		if (x.length != y.length) throw new RuntimeException("Different legnt arrays in subtractArrays().");
		
		double[] z = new double[x.length];
		
		for (int i=0; i<z.length; i++) z[i] = x[i] - y[i];
		
		return z;
	}
	
	
	private void negateArray(double[] x)
	{
		for (int i=0; i<x.length; i++) if (x[i] != 0.0) x[i] = -x[i];
	}

	
	@Override
	public Instances extract_train_features(String train_data_file, String train_rel_file) 
	{
		/*
		 * @TODO: Your code here
		 */
		
		return extract_dataset(train_data_file, train_rel_file).features;
	}

	
	@Override
	public Classifier training(Instances dataset) 
	{
		/*
		 * @TODO: Your code here
		 */
		try 
		{
			model.buildClassifier(dataset);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			throw new RuntimeException("Exception building classifier.", e);
		}
		
		return model;
	}

	
	@Override
	public TestFeatures extract_test_features(String test_data_file) 
	{
		/*
		 * @TODO: Your code here
		 */
		
		return extract_dataset(test_data_file);
	}
	
	
	class PairwiseComparator implements Comparator<String>
	{
		private String query;
		private TestFeatures tf;
		private Classifier model;
		
		PairwiseComparator(String query, TestFeatures testFeatures, Classifier model)
		{
			this.query = query;
			this.tf = testFeatures;
			this.model = model;
		}
		
		@Override
		public int compare(String url1, String url2) 
		{
			Map<String, Integer> urlIndexMap = tf.index_map.get(query);
			
			if (url1.equals(url2)) return 0;
			
			int sign = 1;
			Integer index  = urlIndexMap.get(url1 + " " + url2);
			if (index == null)
			{
				index  = urlIndexMap.get(url2 + " " + url1);
				sign = -1;
			}
			if (index == null) throw new RuntimeException("null index in compare!");
			
			int result;
			try 
			{
				result = (int)model.classifyInstance(tf.features.get(index));
			} 
			catch (Exception e) 
			{
				e.printStackTrace();
				throw new RuntimeException("Exception in classifyInstance.", e);
			}

			return result == 0 ? 1*sign : -1*sign;
		}
	}

	
	@Override
	public Map<String, List<String>> testing(TestFeatures tf, Classifier model) 
	{
		/*
		 * @TODO: Your code here
		 */
		Map<String, List<String>> rankedQueries = new HashMap<String, List<String>>();
		
		for (String query : tf.index_map.keySet())
		{
			TreeSet<String> results = new TreeSet<>(new PairwiseComparator(query, tf, model));
						
			Map<String, Integer> urlIndexMap = tf.index_map.get(query);
			
			for (String url : urlIndexMap.keySet())
			{
				String[] urls = url.split(" ");
				
				results.add(urls[0]);
				results.add(urls[1]);
			}
			
			rankedQueries.put(query, new ArrayList<String>(results));
		}

		return rankedQueries;
	}

}
