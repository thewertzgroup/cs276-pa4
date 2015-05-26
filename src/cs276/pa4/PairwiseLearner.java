package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));

		String[] labels = {"-1", "1"};
		attributes.add(new Attribute("class", Arrays.asList(labels)));
		dataset = new Instances(relevance_file != null ? "train_dataset" : "test_dataset", attributes, 0);
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
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
					
					double[] iInstance = getTFIDFVector(docs.get(i), query);
					double[] jInstance = getTFIDFVector(docs.get(j), query);
					
					double[] instance = subtractArrays(iInstance, jInstance);

					if (relMap != null)
					{
						rel = (int)Math.signum(iRel - jRel);
						if ((rel < 0 && positive < negative) || (rel > 0 && positive > negative))
						{
							rel *= -1;
							negateArray(instance);
						}
						
						if (rel > 0) positive++;
						if (rel < 0) negative++;
						if (rel == 0) throw new RuntimeException("'0' relevance when lable should be '-1' or '1'.");
					}

					Instance inst = new DenseInstance(6);					
				    inst.setDataset(dataset);
				    copyArray(inst, instance);
				    
				    if (relMap != null)
				    {
						String label = Integer.toString(rel);
					    inst.setValue(5, label); 
				    }
				    
					dataset.add(inst);
					
					urlIndexMap.put(docs.get(i).url + " " + docs.get(j).url, dataset.size()-1);
				}
			}
			
			testFeatures.index_map.put(query.query,  urlIndexMap);
		}
		
		System.err.println("\nDataset:\n\n" + dataset);
		
		Standardize filter = new Standardize();	  
		try {
			filter.setInputFormat(dataset);
			testFeatures.features = Filter.useFilter(dataset, filter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.err.println("\nStandardized Dataset:\n\n" + testFeatures.features);
		
		return testFeatures;
	}


	private void copyArray(Instance inst, double[] instance) 
	{
	    for(int i = 0; i < instance.length; i++)
	    {
	    	inst.setValue(i, instance[i]);	
	    }
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

	
	@Override
	public Map<String, List<String>> testing(TestFeatures tf, Classifier model) 
	{
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

}
