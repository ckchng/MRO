
#ifndef ViewDatabase_hpp
#define ViewDatabase_hpp

#include "ViewDatabase.hpp"
#include <stdio.h>
#include "Frame.hpp"
#include "View.hpp"
#include "ORBVocabulary.hpp"


namespace irotavg
{
    class ViewDatabase
    {
    public:
        
        ViewDatabase(ViewDatabase const&) = delete;
        
        void operator=(ViewDatabase const&) = delete;
        
        static ViewDatabase& instance()
        {
            static ViewDatabase instance;
            return instance;
        }
        
        void init()
        {
            const ORBVocabulary &voc = ORBVocabulary::instance();
            m_inverted_file.resize(voc.vocabulary().size());
        }
        
        
        void add(View *view);
        void erase(View *view);
        
        void clear()
        {
            m_inverted_file.clear();
            init();
        }

        std::list<View*> findViewsSharingWords(View &view, std::map<View*, int> &num_of_shared_words);
        std::vector<View*> detectLoopCandidates(View &view, double min_score);
        
      
        
    private:
        
        ViewDatabase(){}
        
        //std::string m_vocabulary_filename;
        
        // Inverted file
        std::vector< std::list<View*> > m_inverted_file;
    };
}


#endif /* ViewDatabase_hpp */
