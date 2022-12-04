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
      d_sEncode = ENCODE_S_IDLE;
      d_debug = true;
      d_pktSeq = 0;
      d_nChipsPadded = 624;   // sometimes num of chips are too small which do not trigger the stream passing

      memset(d_vhtBfQbytesR, 0, 1024);
      memset(d_vhtBfQbytesI, 0, 1024);

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
      /* 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP*/
      pmt::pmt_t vector = pmt::cdr(msg);
      int tmpMsgLen = pmt::blob_length(vector);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(vector, tmpOffset);
      if((tmpMsgLen < 5) || (tmpMsgLen > DECODE_D_MAX)){
        return;
      }

      int tmpFormat = (int)tmpPkt[0];

      if(tmpFormat == C8P_F_VHT_BFQ_R)
      {
        memcpy(d_vhtBfQbytesR, &tmpPkt[1], 1024);
        std::cout<<"beamforming Q real updated"<<std::endl;
        return;
      }

      if(tmpFormat == C8P_F_VHT_BFQ_I)
      {
        memcpy(d_vhtBfQbytesI, &tmpPkt[1], 1024);
        std::cout<<"beamforming Q imag updated"<<std::endl;
        return;
      }

      int tmpHeaderShift;
      
      if(tmpFormat == C8P_F_VHT_MU)
      {
        // byte 0 format, user0 1-4, user1 5-8, group ID 9
        tmpHeaderShift = 10;
        int tmpMcs0 = (int)tmpPkt[1];
        // int tmpNss0 = (int)tmpPkt[2];
        int tmpLen0 = ((int)tmpPkt[4] * 256  + (int)tmpPkt[3]);
        int tmpMcs1 = (int)tmpPkt[5];
        // int tmpNss1 = (int)tmpPkt[6];
        int tmpLen1 = ((int)tmpPkt[8] * 256  + (int)tmpPkt[7]);
        int tmpGroupId = (int)tmpPkt[9];
        dout<<"ieee80211 encode, format: mu, mcs0:"<<tmpMcs0<<", nSS0:1, len0:"<<tmpLen0<<", mcs1:"<<tmpMcs1<<", nSS1:1, len1:"<<tmpLen1<<std::endl;
        formatToModMu(&d_m, tmpMcs0, 1, tmpLen0, tmpMcs1, 1, tmpLen1);
        d_m.groupId = tmpGroupId;
        float* tmpFloatPR = (float*)d_vhtBfQbytesR;
        float* tmpFloatPI = (float*)d_vhtBfQbytesI;
        d_tagBfQ.clear();
        d_tagBfQ.reserve(256);
        gr_complex tmpQValue;
        for(int i=0;i<256;i++)
        {
          tmpQValue = gr_complex(*tmpFloatPR, *tmpFloatPI);
          tmpFloatPR += 1;
          tmpFloatPI += 1;
          d_tagBfQ.push_back(tmpQValue);
        }
      }
      else
      {
        tmpHeaderShift = 5;
        int tmpMcs = (int)tmpPkt[1];
        int tmpNss = (int)tmpPkt[2];
        int tmpLen = ((int)tmpPkt[4] * 256  + (int)tmpPkt[3]);
        dout<<"ieee80211 encode, format:"<<tmpFormat<<", mcs:"<<tmpMcs<<", nSS:"<<tmpNss<<", len:"<<tmpLen<<std::endl;
        formatToModSu(&d_m, tmpFormat, tmpMcs, tmpNss, tmpLen);
      }

      if(d_m.format == C8P_F_L)
      {
        // legacy
        legacySigBitsGen(d_legacySig, d_legacySigCoded, d_m.mcs, d_m.len);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        uint8_t* tmpDataP = d_dataBits;
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));
      }
      else if(d_m.format == C8P_F_VHT)
      {
        // vht
        vhtSigABitsGen(d_vhtSigA, d_vhtSigACoded, &d_m);
        procIntelLegacyBpsk(&d_vhtSigACoded[0], &d_vhtSigAInted[0]);
        procIntelLegacyBpsk(&d_vhtSigACoded[48], &d_vhtSigAInted[48]);
        if(d_m.sumu)
        {
          vhtSigB20BitsGenMU(d_vhtSigB, d_vhtSigBCoded, d_vhtSigBCrc8, d_vhtSigBMu1, d_vhtSigBMu1Coded, d_vhtSigBMu1Crc8, &d_m);
          procIntelVhtB20(d_vhtSigBCoded, d_vhtSigBInted);
          procIntelVhtB20(d_vhtSigBMu1Coded, d_vhtSigBMu1Inted);
        }
        else
        {
          vhtSigB20BitsGenSU(d_vhtSigB, d_vhtSigBCoded, d_vhtSigBCrc8, &d_m);
          procIntelVhtB20(d_vhtSigBCoded, d_vhtSigBInted);
        }

        if(d_m.nSym > 0)
        {
          // legacy training 16, legacy sig 4, vhtsiga 8, vht training 4+4n, vhtsigb, payload, no short GI
          int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4 + d_m.nSym * 4;
          int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
          legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
          procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

          if(d_m.sumu)
          {
            vhtModMuToSu(&d_m, 0);  // set mod info to be user 0
          }

          uint8_t* tmpDataP = d_dataBits;   // data bits, service part
          memset(tmpDataP, 0, 8);
          tmpDataP += 8;
          memcpy(tmpDataP, d_vhtSigBCrc8, 8);
          tmpDataP += 8;
          for(int i=0;i<d_m.len;i++)        // data bits, psdu part
          {
            for(int j=0;j<8;j++)
            {
              tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
            }
            tmpDataP += 8;
          }
          tmpHeaderShift += d_m.len;

          int tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6) / 8;           // 20M 2x2, nES is still 1
          memcpy(tmpDataP, &d_dataBits[16], (tmpPsduLen - d_m.len)*8);    // EOF padding tmp, copy header bits to pad
          tmpDataP += (tmpPsduLen - d_m.len)*8;                           // padded bits, all 0, includes 6 tail bits, when scrambling, do not scramble tail
          memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));

          if(d_m.sumu)
          {
            // set mod info to be user 1, this version no other users
            vhtModMuToSu(&d_m, 1);
            // init pointer
            tmpDataP = d_dataBits2;
            // 7 scrambler init, 1 reserved
            memset(tmpDataP, 0, 8);
            tmpDataP += 8;
            // 8 sig b crc8
            memcpy(tmpDataP, d_vhtSigBMu1Crc8, 8);
            tmpDataP += 8;
            // data
            for(int i=0;i<d_m.len;i++)
            {
              for(int j=0;j<8;j++)
              {
                tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
              }
              tmpDataP += 8;
            }
            // general packet with payload
            tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6)/8;
            // EOF padding tmp, copy header bits to pad
            memcpy(tmpDataP, &d_dataBits[16], (tmpPsduLen - d_m.len)*8);
            tmpDataP += (tmpPsduLen - d_m.len)*8;
            // tail pading, all 0, includes tail bits, when scrambling, do not scramble tail
            memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));

          }
        }
        else
        {
          // NDP channel sounding, legacy, vht sig a, vht training, vht sig b
          int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4;
          int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
          legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
          procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);
        }
      }
      else
      {
        // ht
        htSigBitsGen(d_htSig, d_htSigCoded, &d_m);
        procIntelLegacyBpsk(&d_htSigCoded[0], &d_htSigInted[0]);
        procIntelLegacyBpsk(&d_htSigCoded[48], &d_htSigInted[48]);
        // legacy training and sig 20, htsig 8, ht training 4+4n, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        uint8_t* tmpDataP = d_dataBits;
        // service
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        // data
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));
      }
      d_sEncode = ENCODE_S_SCEDULE;
    }

    int
    encode_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sEncode == ENCODE_S_SCEDULE)
      {
        dout<<"ieee80211 encode, schedule in calculate, nSym:"<<d_m.nSym<<", total sample output:"<<(d_m.nSym * d_m.nSD)<<std::endl;
        d_nChipsGen = d_m.nSym * d_m.nSD;
        d_nChipsGenProcd = 0;
        d_sEncode = ENCODE_S_ENCODE;
      }
      return (d_nChipsGen + d_nChipsPadded);
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
          return 0;
        }

        case ENCODE_S_SCEDULE:
        {
          return 0;
        }

        case ENCODE_S_ENCODE:
        {
          dout<<"ieee80211 encode, encode and gen tag, seq:"<<d_pktSeq<<std::endl;
          if(d_m.sumu)
          {
            // user 0
            vhtModMuToSu(&d_m, 0);
            scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS - 6), 93);
            memset(&d_scramBits[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            // binary convolutional coding
            bccEncoder(d_scramBits, d_convlBits, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits, d_punctBits, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave
            for(int i=0;i<d_m.nSym;i++)
            {
              procInterNonLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m, 0);
            }
            // bits to qam chips
            bitsToChips(d_IntedBits1, d_qamChips1, &d_m);

            // user 1
            vhtModMuToSu(&d_m, 1);
            scramEncoder(d_dataBits2, d_scramBits2, (d_m.nSym * d_m.nDBPS - 6), 93);
            memset(&d_scramBits2[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            // binary convolutional coding
            bccEncoder(d_scramBits2, d_convlBits2, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits2, d_punctBits2, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave
            for(int i=0;i<d_m.nSym;i++)
            {
              procInterNonLegacy(&d_punctBits2[i*d_m.nCBPS], &d_IntedBits2[i*d_m.nCBPS], &d_m, 0);
            }
            // bits to qam chips
            bitsToChips(d_IntedBits2, d_qamChips2, &d_m);
          }
          else if(d_m.nSym > 0)
          {
            // scrambling
            if(d_m.format == C8P_F_VHT)
            {
              scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS - 6), 93);
              memset(&d_scramBits[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            }
            else
            {
              scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS), 93);
              memset(&d_scramBits[d_m.len * 8 + 16], 0, 6);
            }
            // binary convolutional coding
            bccEncoder(d_scramBits, d_convlBits, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits, d_punctBits, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave and convert to qam chips
            if(d_m.nSS == 1)
            {
              if(d_m.format == C8P_F_L)
              {
                for(int i=0;i<d_m.nSym;i++)
                {
                  procInterLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m);
                }
              }
              else
              {
                for(int i=0;i<d_m.nSym;i++)
                {
                  procInterNonLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m, 0);
                }
              }
              bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
              memset(d_qamChips2, 0, d_m.nSym * d_m.nSD);
            }
            else
            {
              // stream parser first
              streamParser2(d_punctBits, d_parsdBits1, d_parsdBits2, d_m.nSym * d_m.nCBPS, &d_m);
              // interleave
              for(int i=0;i<d_m.nSym;i++)
              {
                procInterNonLegacy(&d_parsdBits1[i*d_m.nCBPSS], &d_IntedBits1[i*d_m.nCBPSS], &d_m, 0);  // iss - 1 = 0
                procInterNonLegacy(&d_parsdBits2[i*d_m.nCBPSS], &d_IntedBits2[i*d_m.nCBPSS], &d_m, 1);  // iss - 1 = 1
              }
              // convert to qam chips
              bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
              bitsToChips(d_IntedBits2, d_qamChips2, &d_m);
            }
          }
          else
          {
            // VHT NDP
          }

          // gen tag
          d_tagLegacyBits.clear();
          d_tagLegacyBits.reserve(48);
          for(int i=0;i<48;i++)
          {
            d_tagLegacyBits.push_back(d_legacySigInted[i]);
          }
          pmt::pmt_t dict = pmt::make_dict();
          if(d_m.sumu)
          {
            dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(C8P_F_VHT_MU));
            dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_m.mcsMu[0]));
            dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_m.lenMu[0]));
            dict = pmt::dict_add(dict, pmt::mp("mcs1"), pmt::from_long(d_m.mcsMu[1]));
            dict = pmt::dict_add(dict, pmt::mp("len1"), pmt::from_long(d_m.lenMu[1]));
          }
          else
          {
            dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
            dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
            dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_m.nSS));
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          }
          dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
          dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(d_nChipsGen + d_nChipsPadded));   // chips with padded
          dict = pmt::dict_add(dict, pmt::mp("lsig"), pmt::init_u8vector(d_tagLegacyBits.size(), d_tagLegacyBits));
          d_pktSeq++;
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
            // sig a bits
            d_tagVhtABits.clear();
            d_tagVhtABits.reserve(96);
            for(int i=0;i<96;i++)
            {
              d_tagVhtABits.push_back(d_vhtSigAInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("vhtsiga"), pmt::init_u8vector(d_tagVhtABits.size(), d_tagVhtABits));
            // sig b bits
            d_tagVhtBBits.clear();
            d_tagVhtBBits.reserve(52);
            for(int i=0;i<52;i++)
            {
              d_tagVhtBBits.push_back(d_vhtSigBInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("vhtsigb"), pmt::init_u8vector(d_tagVhtBBits.size(), d_tagVhtBBits));
            // mu-mimo, 2nd sig b
            if(d_m.sumu)
            {
              d_tagVhtBMu1Bits.clear();
              d_tagVhtBMu1Bits.reserve(52);
              for(int i=0;i<52;i++)
              {
                d_tagVhtBMu1Bits.push_back(d_vhtSigBMu1Inted[i]);
              }
              dict = pmt::dict_add(dict, pmt::mp("vhtsigb1"), pmt::init_u8vector(d_tagVhtBMu1Bits.size(), d_tagVhtBMu1Bits));
              dict = pmt::dict_add(dict, pmt::mp("vhtbfq"), pmt::init_c32vector(d_tagBfQ.size(), d_tagBfQ));
            }
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
          if(d_m.len > 0)
          {
            d_sEncode = ENCODE_S_COPY;
          }
          else
          {
            // NDP, skip copy
            d_sEncode = ENCODE_S_PAD;
          }
          return 0;
        }

        case ENCODE_S_COPY:
        {
          int o1 = 0;
          while((o1 + d_m.nSD) < d_nGen)
          {
            memcpy(&outChips1[o1], &d_qamChips1[d_nChipsGenProcd], d_m.nSD);
            memcpy(&outChips2[o1], &d_qamChips2[d_nChipsGenProcd], d_m.nSD);

            o1 += d_m.nSD;
            d_nChipsGenProcd += d_m.nSD;
            if(d_nChipsGenProcd >= d_nChipsGen)
            {
              d_sEncode = ENCODE_S_PAD;
              break;
            }
          }
          return o1;
        }

        case ENCODE_S_PAD:
        {
          if(d_nGen >= d_nChipsPadded)
          {
            memset(outChips1, 0, d_nChipsPadded);
            memset(outChips2, 0, d_nChipsPadded);
            d_sEncode = ENCODE_S_IDLE;
            return d_nChipsPadded;
          }
          return 0;
        }
      }

      // Tell runtime system how many output items we produced.
      d_sEncode = ENCODE_S_IDLE;
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
