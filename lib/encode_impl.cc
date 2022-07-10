/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Encoder of 802.11a/g/n/ac 1x1 and 2x2 payload part
 *     Copyright (C) June 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <gnuradio/io_signature.h>
#include "encode_impl.h"

uint8_t test_legacyPkt[105] = {
  0, 0, 1, 0, 100,     // format 0, mcs 0, nss 1, len 100
  8, 1, 48, 0, 102, 85, 68, 51, 34, 1, 102, 85, 68, 51, 34, 2, 102, 85, 68, 51, 
  34, 1, 192, 2, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 64, 241, 161, 64, 0, 64, 
  17, 173, 182, 192, 168, 13, 3, 192, 168, 13, 1, 140, 72, 34, 185, 0, 44, 22, 
  43, 109, 122, 108, 116, 98, 111, 121, 66, 48, 55, 110, 110, 77, 114, 51, 110, 
  79, 121, 121, 99, 73, 105, 115, 55, 114, 114, 78, 75, 90, 122, 112, 117, 105, 
  78, 79, 116, 251, 174, 122, 183
};

uint8_t test_htPkt[105] = {
  1, 15, 2, 0, 100,
  8, 1, 48, 0, 102, 85, 68, 51, 34, 1, 102, 85, 68, 51, 34, 2, 102, 85, 68, 51, 
  34, 1, 192, 2, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 64, 241, 161, 64, 0, 64, 
  17, 173, 182, 192, 168, 13, 3, 192, 168, 13, 1, 140, 72, 34, 185, 0, 44, 22, 
  43, 109, 122, 108, 116, 98, 111, 121, 66, 48, 55, 110, 110, 77, 114, 51, 110, 
  79, 121, 121, 99, 73, 105, 115, 55, 114, 114, 78, 75, 90, 122, 112, 117, 105, 
  78, 79, 116, 251, 174, 122, 183
};

uint8_t test_vhtPkt[105] = {
  2, 8, 2, 0, 100,
  1, 6, 157, 78, 136, 1, 110, 0, 244, 105, 213, 128, 15, 160, 0, 192, 202, 177, 
  91, 225, 244, 105, 213, 128, 15, 160, 0, 169, 0, 0, 170, 170, 3, 0, 0, 0, 8, 
  0, 69, 0, 0, 58, 171, 2, 64, 0, 64, 17, 123, 150, 10, 10, 0, 6, 10, 10, 0, 1, 
  153, 211, 34, 185, 0, 38, 16, 236, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 49, 
  50, 51, 52, 53, 54, 55, 56, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 
  41, 169, 161, 121
};

namespace gr {
  namespace ieee80211 {

    encode::sptr
    encode::make(const std::string& tsb_tag_key)
    {
      return gnuradio::make_block_sptr<encode_impl>(tsb_tag_key
        );
    }


    /*
     * The private constructor
     */
    encode_impl::encode_impl(const std::string& tsb_tag_key)
      : gr::tagged_stream_block("encode",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(2, 2, sizeof(uint8_t)), tsb_tag_key)
    {
      //message_port_register_in(pdu::pdu_port_id());
      d_sEncode = ENCODE_S_IDLE;

      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&encode_impl::msgRead, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    encode_impl::~encode_impl()
    {
    }

    void
    encode_impl::msgRead(pmt::pmt_t msg)
    {
      std::cout<<"ieee80211 encode, new msg";
      // pmt::pmt_t vector = pmt::cdr(msg);
      // int tmpMsgLen = pmt::blob_length(vector);
      // size_t tmpOffset(0);
      // const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(vector, tmpOffset);
      // 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP
      // if((tmpMsgLen < 5) || (tmpMsgLen > DECODE_D_MAX)){
      //   return;
      // }

      uint8_t* tmpPkt = test_vhtPkt;
      int tmpLen;
      tmpLen = ((int)tmpPkt[3] * 256  + (int)tmpPkt[4]);
      int tmpHenderShift = 5;
      
      std::cout<<"ieee80211 encode, new msg, len:"<<(int)tmpLen<<std::endl;
      formatToModSu(&d_m, (int)tmpPkt[0], (int)tmpPkt[1], (int)tmpPkt[2], tmpLen);
      if(d_m.format == C8P_F_L)
      {
        // legacy
        std::cout<<"ieee80211 encode, legacy packet"<<std::endl;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, d_m.mcs, d_m.len);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);
        
        std::cout<<"ieee80211 encode, legacy sig bits";
        for(int q=0;q<24;q++)
        {
          std::cout<<(int)d_legacySig[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, legacy sig coded bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_legacySigCoded[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, legacy sig inted bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_legacySigInted[q]<<" ";
        }
        std::cout<<std::endl;


        uint8_t* tmpDataP = d_dataBits;
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHenderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));
        
        std::cout<<"ieee80211 encode, legacy data bits";
        for(int q=0;q<d_m.nSym * d_m.nDBPS;q++)
        {
          std::cout<<(int)d_dataBits[q]<<", ";
        }
        std::cout<<std::endl;
      }
      else if(d_m.format == C8P_F_VHT)
      {
        // vht
        vhtSigABitsGenSU(d_vhtSigA, d_vhtSigACoded, &d_m);
        procIntelLegacyBpsk(&d_vhtSigACoded[0], &d_vhtSigAInted[0]);
        procIntelLegacyBpsk(&d_vhtSigACoded[48], &d_vhtSigAInted[48]);
        vhtSigB20BitsGenSU(d_vhtSigB20, d_vhtSigB20Coded, d_vhtSigBCrc8, &d_m);
        procIntelVhtB20(d_vhtSigB20Coded, d_vhtSigB20Inted);

        int tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6)/8;
        // legacy training 16, legacy sig 4, vhtsiga 8, vht training 4+4n, vhtsigb, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        std::cout<<"ieee80211 encode, legacy sig inted bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_legacySigInted[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, vht sig a inted bits";
        for(int q=0;q<96;q++)
        {
          std::cout<<(int)d_vhtSigAInted[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, vht sig b inted bits";
        for(int q=0;q<52;q++)
        {
          std::cout<<(int)d_vhtSigB20Inted[q]<<" ";
        }
        std::cout<<std::endl;



        uint8_t* tmpDataP = d_dataBits;
        // 7 scrambler init, 1 reserved
        memset(tmpDataP, 0, 8);
        tmpDataP += 8;
        // 8 sig b crc8
        memcpy(tmpDataP, d_vhtSigBCrc8, 8);
        tmpDataP += 8;
        // data
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHenderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }

        // // EOF subframe padding
        // for(int i=0;i<(tmpPsduLen/4);i++)
        // {
        //   memcpy(tmpDataP, EOF_PAD_SUBFRAME, 32);
        //   tmpDataP += 32;
        // }
        // // EOF octect padding
        // for(int i=0;i<(tmpPsduLen%4);i++)
        // {
        //   memset(tmpDataP, 0, 8);
        //   tmpDataP += 8;
        // }

        // EOF padding tmp
        memcpy(tmpDataP, &d_dataBits[16], (tmpPsduLen - d_m.len)*8);
        tmpDataP += (tmpPsduLen - d_m.len)*8;

        // tail pading, all 0, includes tail bits, when scrambling, do not scramble tail
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));

        std::cout<<"ieee80211 encode, vht data bits";
        for(int q=0;q<d_m.nSym * d_m.nDBPS;q++)
        {
          std::cout<<(int)d_dataBits[q]<<", ";
        }
        std::cout<<std::endl;
      }
      else
      {
        // ht
        std::cout<<"ieee80211 encode, ht packet"<<std::endl;
        htSigBitsGen(d_htSig, d_htSigCoded, &d_m);
        procIntelLegacyBpsk(&d_htSigCoded[0], &d_htSigInted[0]);
        procIntelLegacyBpsk(&d_htSigCoded[48], &d_htSigInted[48]);
        // legacy training and sig 20, htsig 8, ht training 4+4n, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        std::cout<<"ieee80211 encode, legacy sig bits";
        for(int q=0;q<24;q++)
        {
          std::cout<<(int)d_legacySig[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, legacy sig coded bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_legacySigCoded[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, legacy sig inted bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_legacySigInted[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, ht sig bits";
        for(int q=0;q<48;q++)
        {
          std::cout<<(int)d_htSig[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, ht sig coded bits";
        for(int q=0;q<96;q++)
        {
          std::cout<<(int)d_htSigCoded[q]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"ieee80211 encode, ht sig inted bits";
        for(int q=0;q<96;q++)
        {
          std::cout<<(int)d_htSigInted[q]<<" ";
        }
        std::cout<<std::endl;


        uint8_t* tmpDataP = d_dataBits;
        // service
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        // data
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHenderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));

        std::cout<<"ieee80211 encode, ht data bits";
        for(int q=0;q<d_m.nSym * d_m.nDBPS;q++)
        {
          std::cout<<(int)d_dataBits[q]<<", ";
        }
        std::cout<<std::endl;
      }
      d_sEncode = ENCODE_S_SCEDULE;
    }

    int
    encode_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sEncode == ENCODE_S_SCEDULE)
      {
        std::cout<<"schedule in calculate"<<std::endl;
        d_nChipsGen = d_m.nSym * d_m.nSD;      // gen payload part qam chips
        d_nChipsGenProcd = 0;
        d_sEncode = ENCODE_S_ENCODE;
      }
      return d_nChipsGen;
    }

    int
    encode_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      //auto in = static_cast<const input_type*>(input_items[0]);
      uint8_t* outChips1 = static_cast<uint8_t*>(output_items[0]);
      uint8_t* outChips2 = static_cast<uint8_t*>(output_items[1]);
      d_nGen = noutput_items;

      switch(d_sEncode)
      {
        case ENCODE_S_IDLE:
        {
          std::cout<<"idle"<<std::endl;
          return 0;
        }

        case ENCODE_S_SCEDULE:
        {
          std::cout<<"schedule in work"<<std::endl;
          return 0;
        }

        case ENCODE_S_ENCODE:
        {
          std::cout<<"ieee80211 encode, encode and gen tag"<<std::endl;
          // scrambling
          if(d_m.format == C8P_F_VHT)
          {
            std::cout<<"ieee80211 encode, vht scrambling"<<std::endl;
            scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS - 6), 97);
            memset(&d_scramBits[d_m.nSym * d_m.nDBPS - 6], 0, 6);

            std::cout<<"ieee80211 encode, vht scrambled bits: ";
            for(int q=0;q<d_m.nSym * d_m.nDBPS;q++)
            {
              std::cout<<(int)d_scramBits[q]<<", ";
            }
            std::cout<<std::endl;
          }
          else
          {
            std::cout<<"ieee80211 encode, non-vht scrambling"<<std::endl;
            scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS), 97);
            memset(&d_scramBits[d_m.len * 8 + 16], 0, 6);

            std::cout<<"ieee80211 encode, non-vht scrambled bits: ";
            for(int q=0;q<d_m.nSym * d_m.nDBPS;q++)
            {
              std::cout<<(int)d_scramBits[q]<<", ";
            }
            std::cout<<std::endl;
          }
          // binary convolutional coding
          bccEncoder(d_scramBits, d_convlBits, d_m.nSym * d_m.nDBPS);

          std::cout<<"ieee80211 encode, coded bits: ";
          for(int q=0;q<d_m.nSym * d_m.nDBPS * 2;q++)
          {
            std::cout<<(int)d_convlBits[q]<<", ";
          }
          std::cout<<std::endl;
          
          // puncturing
          punctEncoder(d_convlBits, d_punctBits, d_m.nSym * d_m.nDBPS * 2, &d_m);

          std::cout<<"ieee80211 encode, punctured bits: ";
          for(int q=0;q<d_m.nSym * d_m.nCBPS;q++)
          {
            std::cout<<(int)d_punctBits[q]<<", ";
          }
          std::cout<<std::endl;

          // interleave and convert to qam chips
          if(d_m.nSS == 1)
          {
            if(d_m.format == C8P_F_L)
            {
              for(int i=0;i<d_m.nSym;i++)
              {
                procInterLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m);
              }

              std::cout<<"ieee80211 encode, legacy inteleaved bits: ";
              for(int q=0;q<d_m.nSym * d_m.nCBPS;q++)
              {
                std::cout<<(int)d_IntedBits1[q]<<", ";
              }
              std::cout<<std::endl;
            }
            else
            {
              for(int i=0;i<d_m.nSym;i++)
              {
                procInterNonLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m, 0);
              }

              std::cout<<"ieee80211 encode, non-legacy inteleaved bits: ";
              for(int q=0;q<d_m.nSym * d_m.nCBPS;q++)
              {
                std::cout<<(int)d_IntedBits1[q]<<", ";
              }
              std::cout<<std::endl;

            }
            bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
            memset(d_qamChips2, 0, d_m.nSym * d_m.nSD);

            std::cout<<"ieee80211 encode, single ss chips: ";
            for(int q=0;q<d_m.nSym * d_m.nSD;q++)
            {
              std::cout<<(int)d_qamChips1[q]<<", ";
            }
            std::cout<<std::endl;
          }
          else
          {
            // stream parser first
            streamParser2(d_punctBits, d_parsdBits1, d_parsdBits2, d_m.nSym * d_m.nCBPS, &d_m);

            std::cout<<"ieee80211 encode, parsdBits 1 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nCBPSS;q++)
            {
              std::cout<<(int)d_parsdBits1[q]<<", ";
            }
            std::cout<<std::endl;

            std::cout<<"ieee80211 encode, parsdBits 2 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nCBPSS;q++)
            {
              std::cout<<(int)d_parsdBits2[q]<<", ";
            }
            std::cout<<std::endl;

            for(int i=0;i<d_m.nSym;i++)
            {
              procInterNonLegacy(&d_parsdBits1[i*d_m.nCBPSS], &d_IntedBits1[i*d_m.nCBPSS], &d_m, 0);  // iss - 1 = 0
              procInterNonLegacy(&d_parsdBits2[i*d_m.nCBPSS], &d_IntedBits2[i*d_m.nCBPSS], &d_m, 1); // iss - 1 = 1
            }

            std::cout<<"ieee80211 encode, IntedBits 1 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nCBPSS;q++)
            {
              std::cout<<(int)d_IntedBits1[q]<<", ";
            }
            std::cout<<std::endl;

            std::cout<<"ieee80211 encode, IntedBits 2 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nCBPSS;q++)
            {
              std::cout<<(int)d_IntedBits2[q]<<", ";
            }
            std::cout<<std::endl;

            bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
            bitsToChips(d_IntedBits2, d_qamChips2, &d_m);

            std::cout<<"ieee80211 encode, ss 1 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nSD;q++)
            {
              std::cout<<(int)d_qamChips1[q]<<", ";
            }
            std::cout<<std::endl;

            std::cout<<"ieee80211 encode, ss 2 chips: ";
            for(int q=0;q<d_m.nSym * d_m.nSD;q++)
            {
              std::cout<<(int)d_qamChips2[q]<<", ";
            }
            std::cout<<std::endl;

          }

          // gen tag
          d_tagLegacyBits.clear();
          d_tagLegacyBits.reserve(48);
          for(int i=0;i<48;i++)
          {
            d_tagLegacyBits.push_back(d_legacySigInted[i]);
          }
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
          dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
          dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_m.nSS));
          dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          dict = pmt::dict_add(dict, pmt::mp("lsig"), pmt::init_u8vector(d_tagLegacyBits.size(), d_tagLegacyBits));
          if(d_m.format == C8P_F_HT)
          {
            d_tagHtBits.clear();
            d_tagHtBits.reserve(96);
            for(int i=0;i<96;i++)
            {
              d_tagHtBits.push_back(d_htSigInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("htsig"), pmt::init_u8vector(d_tagHtBits.size(), d_tagHtBits));
          }
          else if(d_m.format == C8P_F_VHT)
          {
            d_tagVhtABits.clear();
            d_tagVhtABits.reserve(96);
            d_tagVhtB20Bits.clear();
            d_tagVhtB20Bits.reserve(52);
            for(int i=0;i<96;i++)
            {
              d_tagVhtABits.push_back(d_vhtSigAInted[i]);
            }
            for(int i=0;i<52;i++)
            {
              d_tagVhtB20Bits.push_back(d_vhtSigB20Inted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("vhtsiga"), pmt::init_u8vector(d_tagVhtABits.size(), d_tagVhtABits));
            dict = pmt::dict_add(dict, pmt::mp("vhtsigb"), pmt::init_u8vector(d_tagVhtB20Bits.size(), d_tagVhtB20Bits));
          }
          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (int i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0,                   // output port index
                            nitems_written(0),  // output sample index
                            pmt::car(pair),     
                            pmt::cdr(pair),
                            alias_pmt());
          }

          d_sEncode = ENCODE_S_COPY;
          return 0;
        }

        case ENCODE_S_COPY:
        {
          std::cout<<"copy"<<std::endl;
          int o1 = 0;
          while((o1 + d_m.nSD) < d_nGen)
          {
            memcpy(&outChips1[o1], &d_qamChips1[d_nChipsGenProcd], d_m.nSD);
            memcpy(&outChips2[o1], &d_qamChips2[d_nChipsGenProcd], d_m.nSD);
            o1 += d_m.nSD;
            d_nChipsGenProcd += d_m.nSD;
            if(d_nChipsGenProcd >= d_nChipsGen)
            {
              std::cout<<"copy done"<<std::endl;
              d_sEncode = ENCODE_S_IDLE;
              break;
            }
          }
          return o1;
        }
      }

      // Tell runtime system how many output items we produced.
      d_sEncode = ENCODE_S_IDLE;
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
