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
              gr::io_signature::make(1, 1, sizeof(uint8_t)), tsb_tag_key)
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
      pmt::pmt_t vector = pmt::cdr(msg);
      int tmpMsgLen = pmt::blob_length(vector);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(vector, tmpOffset);
      uint16_t tmpLen;
      // 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP
      if((tmpMsgLen < 5) || (tmpMsgLen > DECODE_D_MAX)){
        return;
      }
      //memcpy(d_msg, tmpPkt, tmpMsgLen);
      tmpLen = (((uint16_t)tmpPkt[3])<<8  + (uint16_t)tmpPkt[4]);
      formatToModSu(&d_m, (int)tmpPkt[0], (int)tmpPkt[1], (int)tmpPkt[2], (int)tmpLen);
      if(d_m.format == C8P_F_L)
      {
        // legacy
        legacySigBitsGen(d_legacySig, d_legacySigCoded, d_m.mcs, d_m.len);

        uint8_t* tmpDataP = d_dataBits;
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i] >> j);
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
        vhtSigABitsGenSU(d_vhtSigA, d_vhtSigACoded, &d_m);
        vhtSigB20BitsGenSU(d_vhtSigB20, d_vhtSigB20Coded, d_vhtSigBCrc8, &d_m);
        int tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6)/8;
        // legacy training 16, legacy sig 4, vhtsiga 8, vht training 4+4n, vhtsigb, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);

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
            tmpDataP[j] = (tmpPkt[i] >> j);
          }
          tmpDataP += 8;
        }
        // EOF subframe padding
        for(int i=0;i<(tmpPsduLen/4);i++)
        {
          memcpy(tmpDataP, EOF_PAD_SUBFRAME, 32);
          tmpDataP += 32;
        }
        // EOF octect padding
        for(int i=0;i<(tmpPsduLen%4);i++)
        {
          memset(tmpDataP, 0, 8);
          tmpDataP += 8;
        }
        // tail pading, all 0, includes tail bits, when scrambling, do not scramble tail
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));
      }
      else
      {
        // ht
        htSigBitsGen(d_htSig, d_htSigCoded, &d_m);
        // legacy training and sig 20, htsig 8, ht training 4+4n, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);

        uint8_t* tmpDataP = d_dataBits;
        // service
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        // data
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i] >> j);
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
        std::cout<<"schedule in calculate"<<std::endl;
        d_nBitsGen = d_m.nSym * d_m.nCBPS;      // gen payload part coded bits, signal parts in the tag
        d_nBitsGenProcd = 0;
        d_sEncode = ENCODE_S_ENCODE;
      }
      return d_nBitsGen;
    }

    int
    encode_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      //auto in = static_cast<const input_type*>(input_items[0]);
      uint8_t* out = static_cast<uint8_t*>(output_items[0]);
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
          std::cout<<"encode and gen tag"<<std::endl;
          d_sEncode = ENCODE_S_COPY;
          return 0;
        }

        case ENCODE_S_COPY:
        {
          std::cout<<"copy"<<std::endl;
          int o1 = 0;
          int nCBPS = 1000; 
          while((o1 + nCBPS) < d_nGen)
          {
            o1 += nCBPS;
            d_nBitsGenProcd += nCBPS;
            if(d_nBitsGenProcd >= d_nBitsGen)
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
