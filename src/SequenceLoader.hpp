

#ifndef SequenceLoader_hpp
#define SequenceLoader_hpp

#include <stdio.h>
#include <string.h>
#include <boost/filesystem.hpp>


namespace irotavg
{
    class SequenceLoader
    {
    public:

        typedef std::pair<int,boost::filesystem::path> pair_id_path;

        SequenceLoader(std::string path, std::string im_ext, int timestamp_offset=0);
        
        std::vector<pair_id_path>::const_iterator begin() const
        {
            return m_frames.begin();
        }
        
        std::vector<pair_id_path>::const_iterator end() const
        {
            return m_frames.end();
        }

    private:
        std::vector<pair_id_path> m_frames;
    };
}
    
#endif /* SequenceLoader_hpp */
