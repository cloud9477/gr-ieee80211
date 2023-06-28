/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Encoder of 802.11a/g/n/ac 2x2 payload part
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
#include "encode2_impl.h"

namespace gr {
  namespace ieee80211 {

    encode2::sptr
    encode2::make()
    {
      return gnuradio::make_block_sptr<encode2_impl>(
        );
    }


    /*
     * The private constructor
     */
    encode2_impl::encode2_impl()
      : gr::block("encode2",
              gr::io_signature::make(1, 1, sizeof(uint8_t)),
              gr::io_signature::make(2, 2, sizeof(uint8_t)))
    {
      d_sEncode = ENCODE_S_RDTAG;
      d_sigBitsIntedL = std::vector<uint8_t>(48, 0);
      d_sigBitsIntedNL = std::vector<uint8_t>(96, 0);
      d_sigBitsIntedB0 = std::vector<uint8_t>(52, 0);
      d_sigBitsIntedB1 = std::vector<uint8_t>(52, 0);
    }

    /*
     * Our virtual destructor.
     */
    encode2_impl::~encode2_impl()
    {
    }

    void
    encode2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }

    int
    encode2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* inPkt = static_cast<const uint8_t*>(input_items[0]);
      uint8_t* outChips0 = static_cast<uint8_t*>(output_items[0]);
      uint8_t* outChips1 = static_cast<uint8_t*>(output_items[1]);
      d_nProc = ninput_items[0];
      d_nGen = noutput_items;
      d_nUsed = 0;
      d_nPassed = 0;

      if(d_sEncode == ENCODE_S_RDTAG)
      {
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_pktFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(-1)));
          d_pktMcs0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs0"), pmt::from_long(-1)));
          d_pktNss0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss0"), pmt::from_long(-1)));
          d_pktLen0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len0"), pmt::from_long(-1)));
          d_pktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_nPktTotal = d_pktLen0;
          if(d_pktFormat == C8P_F_VHT_MU)
          {
            d_pktMcs1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs1"), pmt::from_long(-1)));
            d_pktNss1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss1"), pmt::from_long(-1)));
            d_pktLen1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len1"), pmt::from_long(-1)));
            d_pktMuGroupId = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("gid"), pmt::from_long(-1)));
            std::cout<<"ieee80211 encode2, mu #"<<d_pktSeq<<", mcs0:"<<d_pktMcs0<<", nss0:"<<d_pktNss0<<", len0:"<<d_pktLen0<<", mcs1:"<<d_pktMcs1<<", nss1:"<<d_pktNss1<<", len1:"<<d_pktLen1<<std::endl;
            d_nPktTotal += d_pktLen1;
            formatToModMu(&d_m, d_pktMcs0, 1, d_pktLen0, d_pktMcs1, 1, d_pktLen1);
            d_m.groupId = d_pktMuGroupId;
          }
          else
          {
            std::cout<<"ieee80211 encode2, su #"<<d_pktSeq<<", format:"<<d_pktFormat<<", mcs:"<<d_pktMcs0<<", nss:"<<d_pktNss0<<", len:"<<d_pktLen0<<std::endl;
          }
          d_nPktRead = 0;
          d_sEncode = ENCODE_S_RDPKT;
        }
      }

      if(d_sEncode == ENCODE_S_RDPKT)
      {
        if(d_nProc >= (d_nPktTotal - d_nPktRead))
        {
          memcpy(d_pkt + d_nPktRead, inPkt, (d_nPktTotal - d_nPktRead));
          d_nUsed += (d_nPktTotal - d_nPktRead);
          d_sEncode = ENCODE_S_MOD;
        }
        else
        {
          memcpy(d_pkt + d_nPktRead, inPkt, d_nProc);
          d_nPktRead += d_nProc;
          d_nUsed += d_nProc;
        }
      }

      if(d_sEncode == ENCODE_S_MOD)
      {
        pmt::pmt_t dict = pmt::make_dict();
        if(d_pktFormat == C8P_F_VHT_MU)
        {
          // write tag
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
          dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_pktMcs0));
          dict = pmt::dict_add(dict, pmt::mp("nss0"), pmt::from_long(d_pktNss0));
          dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_pktLen0));
          dict = pmt::dict_add(dict, pmt::mp("mcs1"), pmt::from_long(d_pktMcs1));
          dict = pmt::dict_add(dict, pmt::mp("nss1"), pmt::from_long(d_pktNss1));
          dict = pmt::dict_add(dict, pmt::mp("len1"), pmt::from_long(d_pktLen1));
          dict = pmt::dict_add(dict, pmt::mp("gid"), pmt::from_long(d_pktMuGroupId));
          dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
        }
        else
        {
          // signal part
          formatToModSu(&d_m, d_pktFormat, d_pktMcs0, d_pktNss0, d_pktLen0);
          if(d_pktFormat == C8P_F_L)
          {
            legacySigBitsGen(d_sigBitsL, d_sigBitsCodedL, d_m.mcs, d_m.len);
            procIntelLegacyBpsk(d_sigBitsCodedL, &d_sigBitsIntedL[0]);
            memset(d_bits0, 0, 16);   // service bits
          }
          else if(d_pktFormat == C8P_F_VHT)
          {
            vhtSigABitsGen(d_sigBitsNL, d_sigBitsCodedNL, &d_m);
            procIntelLegacyBpsk(&d_sigBitsCodedNL[0], &d_sigBitsIntedNL[0]);
            procIntelLegacyBpsk(&d_sigBitsCodedNL[48], &d_sigBitsIntedNL[48]);
            memset(d_bits0, 0, 8);
            vhtSigB20BitsGenSU(d_sigBitsB, d_sigBitsCodedB, &d_bits0[8], &d_m);   // servcie bits sig b crc
            procIntelVhtB20(d_sigBitsCodedB, &d_sigBitsIntedB0[0]);
            dict = pmt::dict_add(dict, pmt::mp("signl"), pmt::init_u8vector(d_sigBitsIntedNL.size(), d_sigBitsIntedNL));
            dict = pmt::dict_add(dict, pmt::mp("sigb0"), pmt::init_u8vector(d_sigBitsIntedB0.size(), d_sigBitsIntedB0));
            // legacy training 16, legacy sig 4, vhtsiga 8, vht training 4+4n, vhtsigb, payload, no short GI
            int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4 + d_m.nSym * 4;
            int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
            legacySigBitsGen(d_sigBitsL, d_sigBitsCodedL, 0, tmpLegacyLen);
            procIntelLegacyBpsk(d_sigBitsCodedL, &d_sigBitsIntedL[0]);
          }
          else
          {
            htSigBitsGen(d_sigBitsNL, d_sigBitsCodedNL, &d_m);
            procIntelLegacyBpsk(&d_sigBitsCodedNL[0], &d_sigBitsIntedNL[0]);
            procIntelLegacyBpsk(&d_sigBitsCodedNL[48], &d_sigBitsIntedNL[48]);
            dict = pmt::dict_add(dict, pmt::mp("signl"), pmt::init_u8vector(d_sigBitsIntedNL.size(), d_sigBitsIntedNL));
            // legacy training and sig 20, htsig 8, ht training 4+4n, payload, no short GI
            int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + d_m.nSym * 4;
            int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
            legacySigBitsGen(d_sigBitsL, d_sigBitsCodedL, 0, tmpLegacyLen);
            procIntelLegacyBpsk(d_sigBitsCodedL, &d_sigBitsIntedL[0]);
            memset(d_bits0, 0, 16); // service bits
          }
          dict = pmt::dict_add(dict, pmt::mp("sigl"), pmt::init_u8vector(d_sigBitsIntedL.size(), d_sigBitsIntedL));

          // psdu
          int tmpDataP = 16;
          for(int i=0;i<d_m.len;i++)
          {
            for(int j=0;j<8;j++)
            {
              d_bits0[tmpDataP] = (d_pkt[i] >> j) & 0x01;
              tmpDataP++;
            }
          }
          if(d_pktFormat == C8P_F_VHT)
          {
            int tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6) / 8;           // 20M 2x2, nES is still 1
            for(int i=0;i<((tmpPsduLen - d_m.len)/4);i++)
            {
              memcpy(&d_bits0[tmpDataP], EOF_PAD_SUBFRAME, sizeof(uint8_t) * 32);     // eof padding
              tmpDataP += 32;
            }
            memset(&d_bits0[tmpDataP], 0, ((tmpPsduLen - d_m.len)%4) * 8 * sizeof(uint8_t));  // padding octets
            tmpDataP += (((tmpPsduLen - d_m.len)%4) * 8);
            memset(&d_bits0[tmpDataP], 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));        // padding bits and tail
            scramEncoder2(d_bits0, (d_m.nSym * d_m.nDBPS - 6), 93);  // scrambling
          }
          else
          {
            memset(&d_bits0[tmpDataP], 0, 6);   // legacy and ht tail
            tmpDataP += 6;
            memset(&d_bits0[tmpDataP], 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));   // legacy and ht pad
            scramEncoder2(d_bits0, (d_m.nSym * d_m.nDBPS), 93);
            memset(&d_bits0[d_m.len * 8 + 16], 0, 6);
          }
          bccEncoder(d_bits0, d_bitsCoded, d_m.nSym * d_m.nDBPS);   // binary convolutional coding
          punctEncoder(d_bitsCoded, d_bitsPunct, d_m.nSym * d_m.nDBPS * 2, &d_m);   // puncturing
          if(d_m.nSS == 1)
          {
            if(d_m.format == C8P_F_L)
            {
              for(int i=0;i<d_m.nSym;i++)
              {
                procSymIntelL2(&d_bitsPunct[i*d_m.nCBPS], &d_bitsInted0[i*d_m.nCBPS], &d_m);
              }
            }
            else
            {
              for(int i=0;i<d_m.nSym;i++)
              {
                procSymIntelNL2SS1(&d_bitsPunct[i*d_m.nCBPS], &d_bitsInted0[i*d_m.nCBPS], &d_m);
              }
            }
            bitsToChips(d_bitsInted0, d_chips0, &d_m);
          }
          else
          {
            // stream parser first
            streamParser2(d_bitsPunct, d_bitsStream0, d_bitsStream1, d_m.nSym * d_m.nCBPS, &d_m);
            // interleave
            for(int i=0;i<d_m.nSym;i++)
            {
              procSymIntelNL2SS1(&d_bitsStream0[i*d_m.nCBPSS], &d_bitsInted0[i*d_m.nCBPSS], &d_m);
              procSymIntelNL2SS2(&d_bitsStream1[i*d_m.nCBPSS], &d_bitsInted1[i*d_m.nCBPSS], &d_m);
            }
            bitsToChips(d_bitsInted0, d_chips0, &d_m);
            bitsToChips(d_bitsInted1, d_chips1, &d_m);
          }
          d_nSampTotal = d_m.nSym * d_m.nSD + ENCODE_GR_PAD;
          d_nSampCopied = 0;

          // write tag
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
          dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_pktMcs0));
          dict = pmt::dict_add(dict, pmt::mp("nss0"), pmt::from_long(d_pktNss0));
          dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_pktLen0));
          dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
        }
        pmt::pmt_t pairs = pmt::dict_items(dict);
        for (size_t i = 0; i < pmt::length(pairs); i++) {
            pmt::pmt_t pair = pmt::nth(i, pairs);
            add_item_tag(0,                   // output port index
                          nitems_written(0),  // output sample index
                          pmt::car(pair),     
                          pmt::cdr(pair),
                          alias_pmt());
        }
        d_sEncode = ENCODE_S_COPY;
      }

      if(d_sEncode == ENCODE_S_COPY)
      {
        if(d_nGen < (d_nSampTotal - d_nSampCopied))
        {
          memcpy(outChips0, d_chips0 + d_nSampCopied, d_nGen * sizeof(uint8_t));
          if(d_m.nSS == 2)
          {
            memcpy(outChips1, d_chips1 + d_nSampCopied, d_nGen * sizeof(uint8_t));
          }

          d_nPassed += d_nGen;
          d_nSampCopied += d_nGen;
        }
        else
        {
          memcpy(outChips0, d_chips0 + d_nSampCopied, (d_nSampTotal - d_nSampCopied) * sizeof(uint8_t));
          if(d_m.nSS == 2)
          {
            memcpy(outChips1, d_chips1 + d_nSampCopied, (d_nSampTotal - d_nSampCopied) * sizeof(uint8_t));
          }
          d_nPassed += (d_nSampTotal - d_nSampCopied);
          d_nSampCopied = d_nSampTotal;
          std::cout<<"ieee80211 encoder2, output sig done #"<<d_pktSeq<<std::endl;
          d_sEncode = ENCODE_S_CLEAN;
        }
      }

      if(d_sEncode == ENCODE_S_CLEAN)
      {
        if(d_nProc >= ENCODE_GR_PAD)
        {
          d_nUsed += ENCODE_GR_PAD;
          d_sEncode = ENCODE_S_RDTAG;
        }
      }

      consume_each (d_nUsed);
      return d_nPassed;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
