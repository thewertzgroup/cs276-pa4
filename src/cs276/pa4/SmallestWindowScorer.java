package cs276.pa4;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 */
//public class SmallestWindowScorer extends BM25Scorer {
public class SmallestWindowScorer extends AScorer {

	/////// Smallest window specific hyper-parameters ////////
	double B = 2;
	

	//////////////////////////////
	
	Map<Pair<Document, Query>,Double> docQuerySmallestWindow; 	// store the smallest window for a query and a document
	
	// for CosineSimilarityScorer extension
	Map<Query,List<Document>> queryDict; // query -> url -> document
	public SmallestWindowScorer(Map<String, Double> idfs, Map<Query,List<Document>> queryDict) {
		super(idfs);
		this.queryDict = queryDict;
		handleSmallestWindow();		
	}

		
	public void handleSmallestWindow() {
		/*
		 * @//TODO : Your code here
		 */
		Document doc; 
		double curSmallestWindow = -1;
		List<String> docStrList; 
		boolean isBodyField = false; 
		docQuerySmallestWindow = new HashMap<Pair<Document, Query>,Double>(); 
		for(Query q: queryDict.keySet())
		{ 			
			for(int indDoc = 0; indDoc < queryDict.get(q).size(); indDoc++)
			{				 
				doc = queryDict.get(q).get(indDoc); 
				curSmallestWindow= -1;
				isBodyField = false;
				
				docStrList = parseURL(doc.url); 
				curSmallestWindow = checkWindow(q,docStrList, curSmallestWindow, isBodyField); 
				
				if(doc.title!=null)
				{
					docStrList = parseTitle(doc.title); 
					curSmallestWindow = checkWindow(q,docStrList, curSmallestWindow, isBodyField);
				} 
								
				if(doc.headers!=null)
				{ 				 
					for(int headerInd = 0; headerInd < doc.headers.size(); headerInd++)
					{ 
						docStrList = Arrays.asList(doc.headers.get(headerInd).trim().toLowerCase().split("\\+"));  
						curSmallestWindow = checkWindow(q,docStrList, curSmallestWindow, isBodyField);
					} 
				}
				
				if(doc.anchors!=null)
				{ 
					Map<List<String>, Integer> anchorTermsMul = parseAnchors(doc.anchors); 
					for(List<String> anchorList:anchorTermsMul.keySet())
						curSmallestWindow = checkWindow(q,anchorList, curSmallestWindow, isBodyField);
				} 
				
				if(doc.body_hits!=null)
				{ 
					isBodyField = true; 
					docStrList = new ArrayList<String>(); 
					docStrList.add(Integer.toString(indDoc)); 
					curSmallestWindow = checkWindow(q,docStrList, curSmallestWindow, isBodyField);
				} 
		//		if(curSmallestWindow!=-1)
		//			System.out.println(curSmallestWindow); 
				docQuerySmallestWindow.put(new Pair<Document, Query>(doc, q), curSmallestWindow); 
				
			}
		}
		
	}

	public Map<String, List<Integer>> generateQueryHits(Set<String> uniqueQueryWords, List<String> docStrList)
	{ 
		Map<String, List<Integer>> hits = new HashMap<String, List<Integer>>();
			
		if(uniqueQueryWords == null)
			return hits; 
				
		if(uniqueQueryWords.size()<=docStrList.size())
		{ 
			List<Integer> positions; 
			for(String term: uniqueQueryWords)
			{ 
				for(int ind = 0; ind < docStrList.size(); ind++)
					if(docStrList.get(ind).equals(term)) // for every term that is in the docStrList, the list 
														 // of positions should be sorted in ascending order
					{ 
						if(hits.containsKey(term))
						{ 
							hits.get(term).add(ind);  								
						} 
						else 
						{ 
							positions = new ArrayList<Integer>(); 
							positions.add(ind); 
							hits.put(term, positions);
						}
					} 
							
			}
		} 
		
		 
		return hits; 
		
	}
	
	public double computeSmallestWindow(Map<String, List<Integer>> hits)
	{ 
		double smallestWindow = -1;	
		double currWindow = -1; 
		Map.Entry<String, Integer> minPosTerm, maxPosTerm; 
		String termToMove; 
		int indToMove; 
		Map<String, Integer> currentParsedPositions = new HashMap<String, Integer> ();
		for(String term: hits.keySet())
			currentParsedPositions.put(term,hits.get(term).get(0));
		
		while(true)
		{ 
			List<Map.Entry<String, Integer>> listForSort = 
					new LinkedList<Map.Entry<String, Integer>>(currentParsedPositions.entrySet());
			minPosTerm = (Map.Entry<String, Integer>) Collections.min(listForSort, new Comparator<Map.Entry<String, Integer>>() {
				@Override
				public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
					 return o1.getValue().compareTo(o2.getValue());  
					
				}	
			});
			
			maxPosTerm = (Map.Entry<String, Integer>) Collections.max(listForSort, new Comparator<Map.Entry<String, Integer>>() {
				@Override
				public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
					 return o1.getValue().compareTo(o2.getValue());  
					
				}	
			});
			
			currWindow =  (double) ( maxPosTerm.getValue() -  minPosTerm.getValue() + 1 );
			if(smallestWindow == -1 || currWindow < smallestWindow)
				smallestWindow = currWindow; 
			
			// progressing to the next position in the document, for the term that is earliest in the window.  
			termToMove = minPosTerm.getKey();
			indToMove  =  minPosTerm.getValue();
			indToMove++; 
			if(indToMove < hits.get(termToMove).size())
				currentParsedPositions.put(termToMove,hits.get(termToMove).get(indToMove)); // overwrite older position for this term 
																							// advance it to next position
			else // no more occurences of the leftmost term
				break; 
		} 
		
		return smallestWindow; 
	}
	
	public double checkWindow(Query q,List<String> docStrList,double curSmallestWindow,boolean isBodyField) {
		/*
		 * @//TODO : Your code here
		 */
		Set<String> uniqueQueryWords = new HashSet<String>(q.words);
		int numUniqueQueryWords = uniqueQueryWords.size(); 
		
		double newSmallestWindow = -1; 
		if(!isBodyField)
		{ 
			//..
			Map<String, List<Integer>> hits = generateQueryHits(uniqueQueryWords, docStrList); // term -> [list of positions]
			if(numUniqueQueryWords <= hits.size())
			{ 
			//	System.out.println("looking..."); 
				newSmallestWindow = computeSmallestWindow(hits);
			//	System.out.println("Smallest: " + newSmallestWindow);
			} 
		} 
		else 
		{ 
			String indDocStr = docStrList.get(0);
			int indDoc = Integer.parseInt(indDocStr); 
			if(numUniqueQueryWords <= queryDict.get(q).get(indDoc).body_hits.size() )
			{ 
			//	System.out.println("looking in body..."); 
				newSmallestWindow = computeSmallestWindow(queryDict.get(q).get(indDoc).body_hits);
			//	System.out.println("In body, Smallest: " + newSmallestWindow);
			} 
		} 
		
		if(curSmallestWindow == -1)
			return newSmallestWindow; 
		else if(newSmallestWindow == -1 || newSmallestWindow >= curSmallestWindow)
			return curSmallestWindow;
		else 
			return newSmallestWindow; 
	}
	
	
	@Override
	public Map<String, Double> getFeatures(Document d, Query q) {
		
		Map<String, Double> featureVec = new HashMap<String, Double>();
		double smallestWindow = docQuerySmallestWindow.get(new Pair<Document, Query>(d, q));
		featureVec.put("smallWindow", smallestWindow); 		
		return featureVec;
	} 
	
	@Override
	// dumm, will never be called
	public Map<String, Double> getMoreFeatures(Document d, Query q)
	{ 
		Map<String, Double> dumReturn = null; 
		return dumReturn; 
	} 
	

	
}

