package cs276.pa4;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 */
public class SmallestWindowScorer extends BM25Scorer 
{

	/////// Smallest window specific hyper-parameters ////////
	double B = 2.0;    	    
	double boostmod = -1;

	//////////////////////////////
	Map<Query, Map<Document, Double>> smallestWindows;
	
	public SmallestWindowScorer(Map<String,Double> idfs, Map<Query,List<Document>> queryDict) 
	{
		super(idfs, queryDict);
		handleSmallestWindow();
	}

	
	public void handleSmallestWindow() 
	{
		/*
		 * @//TODO : Your code here
		 */
		smallestWindows = new HashMap<>();

		for (Query query : queryDict.keySet())
		{
			Map<Document, Double> querySmallestWindows = new HashMap<>();
			for (Document doc : queryDict.get(query))
			{
				double smallestWindow = Double.MAX_VALUE;
				
				// "url","title","body","header","anchor"
				smallestWindow = checkWindow(query, getIndexMap(getURLTerms(doc)), smallestWindow);				
				
				smallestWindow = checkWindow(query, getIndexMap(getTitleTerms(doc)), smallestWindow);				
				
				if (doc.body_hits != null)
				{
					Set<String> queryTerms = new HashSet<String>(query.words);
					Set<String> bodyTerms = new HashSet<String>(doc.body_hits.keySet());
					if (bodyTerms.containsAll(queryTerms))
					{
						SortedMap<Integer, String> indexMap = new TreeMap<>();
						for (String term : doc.body_hits.keySet())
						{
							for (Integer hit : doc.body_hits.get(term))
							{
								if (queryTerms.contains(term)) indexMap.put(hit, term);
							}
						}
						smallestWindow = checkWindow(query, indexMap, smallestWindow);
					}
				}
				
				if (doc.headers != null)
				{
					for (String header : doc.headers)
					{
						smallestWindow = checkWindow(query, getIndexMap(getHeaderTerms(header)), smallestWindow);	
					}
				}
				
				if (doc.anchors != null)
				{
					for (String anchor : doc.anchors.keySet())
					{
						smallestWindow = checkWindow(query, getIndexMap(getAnchorTerms(anchor)), smallestWindow);
					}
				}
				
				querySmallestWindows.put(doc, smallestWindow);
			}
			smallestWindows.put(query, querySmallestWindows);
		}
	}
	
	
	SortedMap<Integer, String> getIndexMap(String[] terms)
	{
		int index = 0;
		SortedMap<Integer, String> indexMap = new TreeMap<>();
		for (String term : terms)
		{
			indexMap.put(index++,  term);
		}
		return indexMap;
	}

	
	public double checkWindow(Query q, SortedMap<Integer, String> indexMap, double curSmallestWindow) 
	{
		/*
		 * @//TODO : Your code here
		 */
		double smallestWindow = Double.MAX_VALUE;
		
		Set<String> queryTerms = new HashSet<>(q.words);
		
		while (indexMap.values().containsAll(queryTerms))
		{			
			int start = Integer.MAX_VALUE;
			int end = Integer.MIN_VALUE;
			for (int hit : indexMap.keySet())
			{
				String term = indexMap.get(hit);
				if (queryTerms.contains(term))
				{
					if (hit < start) start = hit;
					if (hit > end) end = hit;
					queryTerms.remove(term);
				}
				if (queryTerms.isEmpty()) break;
			}
			smallestWindow = end - start < smallestWindow ? end - start : smallestWindow;
			
			curSmallestWindow = smallestWindow < curSmallestWindow ? smallestWindow : curSmallestWindow;
			
			queryTerms = new HashSet<>(q.words);
			indexMap.remove(start);
		}
		
		return curSmallestWindow;
	}
	
	
	@Override
	public double getSimScore(Document d, Query q) 
	{
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		/*
		 * If wq,d = ∞, then the boost is 1.
		 * If wq,d = |Q| where Q are the unique terms in q, then we multiply the score
		 * by some factor B.
		 * For values of wq,d between the query length and infinite, we provide a boost
		 * between B and 1. The boost should decrease rapidly with the size of wq,d
		 * and can decrease exponentially or as 1/x.
		 */
		double score = 0.0;
		double BM25Score = super.getSimScore(d, q);
		double smallestWindow = smallestWindows.get(q).get(d);
		
		double querySize = new HashSet<String>(q.words).size();
		
		score = BM25Score * (1 + querySize/smallestWindow);

		return score;
	}

}
