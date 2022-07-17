/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation and OFDM
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
#include "modulation_impl.h"

namespace gr {
  namespace ieee80211 {

    modulation::sptr
    modulation::make()
    {
      return gnuradio::make_block_sptr<modulation_impl>(
        );
    }


    /*
     * The private constructor
     */
    modulation_impl::modulation_impl()
      : gr::block("modulation",
              gr::io_signature::make(2, 2, sizeof(uint8_t)),
              gr::io_signature::make(2, 2, sizeof(gr_complex))),
              d_ofdm_ifft(64,1)
    {
      d_sModul = MODUL_S_IDLE;
      d_debug = true;

      // prepare training fields
      gr_complex tmpSig[64];
      // legacy stf and non legacy stf
      ifft(C8P_STF_F, tmpSig);
      memcpy(&d_stf_l[0], &tmpSig[32], 32*sizeof(gr_complex));
      memcpy(&d_stf_l[32], &tmpSig[0], 64*sizeof(gr_complex));
      memcpy(&d_stf_l[96], &tmpSig[0], 64*sizeof(gr_complex));
      memcpy(&d_stf_nl[0], &tmpSig[48], 16*sizeof(gr_complex));
      memcpy(&d_stf_nl[16], &tmpSig[0], 64*sizeof(gr_complex));
      // legacy stf and non legacy stf with csd for 2nd stream
      memcpy(tmpSig, C8P_STF_F, 64*sizeof(gr_complex));
      procCSD(tmpSig, -200);
      ifft(tmpSig, &d_stf_l2[96]);
      memcpy(&d_stf_l2[0], &d_stf_l2[96], 32*sizeof(gr_complex));
      memcpy(&d_stf_l2[32], &d_stf_l2[96], 64*sizeof(gr_complex));
      memcpy(tmpSig, C8P_STF_F, 64*sizeof(gr_complex));
      procCSD(tmpSig, -400);
      ifft(tmpSig, &d_stf_nl2[16]);
      memcpy(&d_stf_nl2[0], &d_stf_l2[64], 16*sizeof(gr_complex));
      // legacy ltf
      ifft(C8P_LTF_L_F, tmpSig);
      memcpy(&d_ltf_l[0], &tmpSig[32], 32*sizeof(gr_complex));
      memcpy(&d_ltf_l[32], &tmpSig[0], 64*sizeof(gr_complex));
      memcpy(&d_ltf_l[96], &tmpSig[0], 64*sizeof(gr_complex));
      // legaycy ltf with csd
      memcpy(tmpSig, C8P_LTF_L_F, 64*sizeof(gr_complex));
      procCSD(tmpSig, -200);
      ifft(tmpSig, &d_ltf_l2[32]);
      memcpy(&d_ltf_l2[0], &d_ltf_l2[64], 32*sizeof(gr_complex));
      memcpy(&d_ltf_l2[96], &d_ltf_l2[32], 64*sizeof(gr_complex));
      // non legacy ltf
      ifft(C8P_LTF_NL_F, tmpSig);
      memcpy(&d_ltf_nl[0], &tmpSig[48], 16*sizeof(gr_complex));
      memcpy(&d_ltf_nl[16], &tmpSig[0], 64*sizeof(gr_complex));
      // non legaycy ltf with csd
      memcpy(tmpSig, C8P_LTF_NL_F, 64*sizeof(gr_complex));
      procCSD(tmpSig, -400);
      ifft(tmpSig, &d_ltf_nl2[16]);
      memcpy(&d_ltf_nl2[0], &d_ltf_nl2[64], 16*sizeof(gr_complex));
      // non legacy ltf negative
      ifft(C8P_LTF_NL_F_N, tmpSig);
      memcpy(&d_ltf_nl_n[0], &tmpSig[48], 16*sizeof(gr_complex));
      memcpy(&d_ltf_nl_n[16], &tmpSig[0], 64*sizeof(gr_complex));
      // non legacy ltf, vht ss 2 2nd ltf, due to different pilots polarity
      memcpy(tmpSig, C8P_LTF_NL_F_VHT22, 64*sizeof(gr_complex));
      procCSD(tmpSig, -400);
      ifft(tmpSig, &d_ltf_nl_vht22[16]);
      memcpy(&d_ltf_nl_vht22[0], &d_ltf_nl_vht22[64], 16*sizeof(gr_complex));
    }

    /*
     * Our virtual destructor.
     */
    modulation_impl::~modulation_impl()
    {
    }

    void
    modulation_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items + 160;
      ninput_items_required[1] = noutput_items + 160;
    }

    int
    modulation_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* inBits1 = static_cast<const uint8_t*>(input_items[0]);
      const uint8_t* inBits2 = static_cast<const uint8_t*>(input_items[1]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig2 = static_cast<gr_complex*>(output_items[1]);

      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;

      switch(d_sModul)
      {
        case MODUL_S_IDLE:
        {
          get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
          if (d_tags.size())
          {
            dout<<"ieee80211 modulation, got tag."<<std::endl;
            d_sModul = MODUL_S_RD_TAG;
          }
          consume_each(0);
          return 0;
        }
        
        case MODUL_S_RD_TAG:
        {
          int tmpTagFormat, tmpTagMcs, tmpTagNss, tmpTagLen, tmpTagSeq;
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          // basic
          tmpTagFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(9999)));
          tmpTagMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(9999)));
          tmpTagNss = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss"), pmt::from_long(9999)));
          tmpTagLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(9999)));
          tmpTagSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(9999)));
          d_nChipsTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("total"), pmt::from_long(9999)));
          formatToModSu(&d_m, tmpTagFormat, tmpTagMcs, tmpTagNss, tmpTagLen);
          // sig
          d_tagLegacyBits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("lsig"), pmt::PMT_NIL));
          std::copy(d_tagLegacyBits.begin(), d_tagLegacyBits.end(), d_legacySigInted);
          if(tmpTagFormat == C8P_F_VHT)
          {
            d_tagVhtABits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtsiga"), pmt::PMT_NIL));
            std::copy(d_tagVhtABits.begin(), d_tagVhtABits.end(), d_vhtSigAInted);
            d_tagVhtB20Bits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtsigb"), pmt::PMT_NIL));
            std::copy(d_tagVhtB20Bits.begin(), d_tagVhtB20Bits.end(), d_vhtSigB20Inted);
          }
          else if(tmpTagFormat == C8P_F_HT)
          {
            d_tagHtBits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("htsig"), pmt::PMT_NIL));
            std::copy(d_tagHtBits.begin(), d_tagHtBits.end(), d_htSigInted);
          }
          dout<<"ieee80211 modulation, tag read, seq:"<<tmpTagSeq<<std::endl;
          d_sModul = MODUL_S_SIG;
          consume_each(0);
          return 0;
        }

        case MODUL_S_SIG:
        {
          d_sModul = MODUL_S_DATA;
          d_sigPtr1 = d_sig1;
          d_sigPtr2 = d_sig2;
          gr_complex tmpSig1[64];
          gr_complex tmpSig2[64];
          gr_complex tmpSigPilots[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
          d_nSymRd = 0;


          // ss 1 part
          dout<<"ieee80211 modulation, ss 1 signal prepare."<<std::endl;
          // sig gap at begining
          memset(d_sigPtr1, 0, sizeof(gr_complex) * 1000);
          d_sigPtr1 += 1000;
          // legacy training
          memcpy(d_sigPtr1, d_stf_l, sizeof(gr_complex) * 160);
          procToneScaling(d_sigPtr1, 12, d_m.nSS, 160);
          d_sigPtr1 += 160;
          memcpy(d_sigPtr1, d_ltf_l, sizeof(gr_complex) * 160);
          procToneScaling(d_sigPtr1, 52, d_m.nSS, 160);
          d_sigPtr1 += 160;
          // legacy signal
          procChipsToQam(d_legacySigInted, tmpSig1, C8P_QAM_BPSK, 48);
          procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
          procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
          ifft(tmpSig1, tmpSig2);
          procToneScaling(tmpSig2, 52, d_m.nSS, 64);
          memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
          d_sigPtr1 += 16;
          memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
          d_sigPtr1 += 64;
          dout<<"ieee80211 modulation, ss 1 signal done."<<std::endl;

          if(d_m.format == C8P_F_L)
          {
            dout<<"ieee80211 modulation, legacy pkt, go to data."<<std::endl;
            // set pilots for legacy
            d_pilots1[0] = gr_complex(1.0f, 0.0f);
            d_pilots1[1] = gr_complex(1.0f, 0.0f);
            d_pilots1[2] = gr_complex(1.0f, 0.0f);
            d_pilots1[3] = gr_complex(-1.0f, 0.0f);
            d_pilotP = 1;
            // tail pad
            memset(d_sigPtr1, 0, 1000 * sizeof(gr_complex));
            consume_each(0);
            return 0;
          }
          else if(d_m.format == C8P_F_VHT)
          {
            dout<<"ieee80211 modulation, vht pkt, go to data."<<std::endl;
            // vht sig a sym 1
            procChipsToQam(d_vhtSigAInted, tmpSig1, C8P_QAM_BPSK, 48);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 52, d_m.nSS, 64);
            memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr1 += 16;
            memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr1 += 64;
            // vht sig a sym 2
            procChipsToQam(&d_vhtSigAInted[48], tmpSig1, C8P_QAM_QBPSK, 48);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 52, d_m.nSS, 64);
            memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr1 += 16;
            memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr1 += 64;

            // set pilots for vht
            d_pilots1[0] = gr_complex(1.0f, 0.0f);
            d_pilots1[1] = gr_complex(1.0f, 0.0f);
            d_pilots1[2] = gr_complex(1.0f, 0.0f);
            d_pilots1[3] = gr_complex(-1.0f, 0.0f);
            d_pilots2[0] = gr_complex(1.0f, 0.0f);
            d_pilots2[1] = gr_complex(1.0f, 0.0f);
            d_pilots2[2] = gr_complex(1.0f, 0.0f);
            d_pilots2[3] = gr_complex(-1.0f, 0.0f);
            d_pilotP = 4;
          }
          else if(d_m.format == C8P_F_HT)
          {
            dout<<"ieee80211 modulation, ht pkt, go to data."<<std::endl;
            // ht sig sym 1
            procChipsToQam(d_htSigInted, tmpSig1, C8P_QAM_QBPSK, 48);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 52, d_m.nSS, 64);
            memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr1 += 16;
            memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr1 += 64;
            // ht sig sym 2
            procChipsToQam(&d_htSigInted[48], tmpSig1, C8P_QAM_QBPSK, 48);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 52, d_m.nSS, 64);
            memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr1 += 16;
            memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr1 += 64;

            // set pilots for ht
            if(d_m.nSS == 1)
            {
              d_pilots1[0] = gr_complex(1.0f, 0.0f);
              d_pilots1[1] = gr_complex(1.0f, 0.0f);
              d_pilots1[2] = gr_complex(1.0f, 0.0f);
              d_pilots1[3] = gr_complex(-1.0f, 0.0f);
            }
            else
            {
              d_pilots1[0] = gr_complex(1.0f, 0.0f);
              d_pilots1[1] = gr_complex(1.0f, 0.0f);
              d_pilots1[2] = gr_complex(-1.0f, 0.0f);
              d_pilots1[3] = gr_complex(-1.0f, 0.0f);
              d_pilots2[0] = gr_complex(1.0f, 0.0f);
              d_pilots2[1] = gr_complex(-1.0f, 0.0f);
              d_pilots2[2] = gr_complex(-1.0f, 0.0f);
              d_pilots2[3] = gr_complex(1.0f, 0.0f);
            }
            d_pilotP = 3;
          }
          // non-legacy stf
          memcpy(d_sigPtr1, d_stf_nl, sizeof(gr_complex) * 80);
          procToneScaling(d_sigPtr1, 12, d_m.nSS, 80);
          d_sigPtr1 += 80;
          // non-legacy ltf
          memcpy(d_sigPtr1, d_ltf_nl, sizeof(gr_complex) * 80);
          procToneScaling(d_sigPtr1, 56, d_m.nSS, 80);
          d_sigPtr1 += 80;
          if(d_m.nSS == 2)
          {
            memcpy(d_sigPtr1, d_ltf_nl_n, sizeof(gr_complex) * 80);
            procToneScaling(d_sigPtr1, 56, d_m.nSS, 80);
            d_sigPtr1 += 80;
          }
          // vht sig b
          if(d_m.format == C8P_F_VHT)
          {
            procChipsToQam(d_vhtSigB20Inted, tmpSig1, C8P_QAM_BPSK, 52);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_VHT);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_VHT);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 56, d_m.nSS, 64);
            memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr1 += 16;
            memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr1 += 64;
          }


          // ss 2 part
          if(d_m.nSS == 2)
          {
            dout<<"ieee80211 modulation, ss 2 signal prepare."<<std::endl;
            // sig gap at begining
            memset(d_sigPtr2, 0, sizeof(gr_complex) * 1000);
            d_sigPtr2 += 1000;
            // legacy training
            memcpy(d_sigPtr2, d_stf_l2, sizeof(gr_complex) * 160);
            procToneScaling(d_sigPtr2, 12, d_m.nSS, 160);
            d_sigPtr2 += 160;
            memcpy(d_sigPtr2, d_ltf_l2, sizeof(gr_complex) * 160);
            procToneScaling(d_sigPtr2, 52, d_m.nSS, 160);
            d_sigPtr2 += 160;
            // legacy signal
            procChipsToQam(d_legacySigInted, tmpSig1, C8P_QAM_BPSK, 48);
            procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
            procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
            procCSD(tmpSig1, -200);
            ifft(tmpSig1, tmpSig2);
            procToneScaling(tmpSig2, 52, d_m.nSS, 64);
            memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
            d_sigPtr2 += 16;
            memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
            d_sigPtr2 += 64;
            dout<<"ieee80211 modulation, ss 2 signal done."<<std::endl;

            if(d_m.format == C8P_F_VHT)
            {
              dout<<"ieee80211 modulation, vht sig a ss 2."<<std::endl;
              // vht sig a sym 1
              procChipsToQam(d_vhtSigAInted, tmpSig1, C8P_QAM_BPSK, 48);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
              procCSD(tmpSig1, -200);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 52, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
              // vht sig a sym 2
              procChipsToQam(&d_vhtSigAInted[48], tmpSig1, C8P_QAM_QBPSK, 48);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
              procCSD(tmpSig1, -200);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 52, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
            }
            else if(d_m.format == C8P_F_HT)
            {
              dout<<"ieee80211 modulation, ht sig ss 2."<<std::endl;
              // ht sig sym 1
              procChipsToQam(d_htSigInted, tmpSig1, C8P_QAM_QBPSK, 48);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
              procCSD(tmpSig1, -200);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 52, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
              // ht sig sym 2
              procChipsToQam(&d_htSigInted[48], tmpSig1, C8P_QAM_QBPSK, 48);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_L);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
              procCSD(tmpSig1, -200);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 52, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
            }
            // non-legacy stf
            memcpy(d_sigPtr2, d_stf_nl2, sizeof(gr_complex) * 80);
            procToneScaling(d_sigPtr2, 12, d_m.nSS, 80);
            d_sigPtr2 += 80;
            // non-legacy ltf
            memcpy(d_sigPtr2, d_ltf_nl2, sizeof(gr_complex) * 80);
            procToneScaling(d_sigPtr2, 56, d_m.nSS, 80);
            d_sigPtr2 += 80;
            // non-legacy ltf 2nd, VHT has different pilots polarity
            if(d_m.format == C8P_F_VHT)
            {
              memcpy(d_sigPtr2, d_ltf_nl_vht22, sizeof(gr_complex) * 80);
            }
            else
            {
              memcpy(d_sigPtr2, d_ltf_nl2, sizeof(gr_complex) * 80);
            }
            procToneScaling(d_sigPtr2, 56, d_m.nSS, 80);
            d_sigPtr2 += 80;
            // vht sig b
            if(d_m.format == C8P_F_VHT)
            {
              procChipsToQam(d_vhtSigB20Inted, tmpSig1, C8P_QAM_BPSK, 52);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_VHT);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_VHT);
              procCSD(tmpSig1, -400);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 56, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
            }
          }

          consume_each(0);
          return 0;
        }

        case MODUL_S_DATA:
        {
          gr_complex tmpSig1[64];
          gr_complex tmpSig2[64];
          int i1 = 0;
          while((i1 + d_m.nSD) <= d_nProc)
          {
            if(d_m.len > 0)
            {
              if(d_m.format == C8P_F_L)
              {
                d_pilotsTmp[0] = d_pilots1[0] * PILOT_P[d_pilotP];
                d_pilotsTmp[1] = d_pilots1[1] * PILOT_P[d_pilotP];
                d_pilotsTmp[2] = d_pilots1[2] * PILOT_P[d_pilotP];
                d_pilotsTmp[3] = d_pilots1[3] * PILOT_P[d_pilotP];
                d_pilotP = (d_pilotP + 1) % 127;

                procChipsToQam(&inBits1[i1], tmpSig1, d_m.mod, 48);
                procInsertPilotsDc(tmpSig1, tmpSig2, d_pilotsTmp, C8P_F_L);
                procNonDataSc(tmpSig2, tmpSig1, C8P_F_L);
                ifft(tmpSig1, tmpSig2);
                procToneScaling(tmpSig2, 52, d_m.nSS, 64);
                memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
                d_sigPtr1 += 16;
                memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
                d_sigPtr1 += 64;
                dout<<"ieee80211 modulation, gen legacy data sym: "<<d_nSymRd<<", total: "<<d_m.nSym<<std::endl;
              }
              else
              {
                d_pilotsTmp[0] = d_pilots1[0] * PILOT_P[d_pilotP];
                d_pilotsTmp[1] = d_pilots1[1] * PILOT_P[d_pilotP];
                d_pilotsTmp[2] = d_pilots1[2] * PILOT_P[d_pilotP];
                d_pilotsTmp[3] = d_pilots1[3] * PILOT_P[d_pilotP];
                
                procChipsToQam(&inBits1[i1], tmpSig1, d_m.mod, 52);
                procInsertPilotsDc(tmpSig1, tmpSig2, d_pilotsTmp, d_m.format);
                procNonDataSc(tmpSig2, tmpSig1, d_m.format);
                ifft(tmpSig1, tmpSig2);
                procToneScaling(tmpSig2, 56, d_m.nSS, 64);
                memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
                d_sigPtr1 += 16;
                memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
                d_sigPtr1 += 64;
                dout<<"ieee80211 modulation, gen non-legacy data sym: "<<d_nSymRd<<", total: "<<d_m.nSym<<std::endl;
                if(d_m.nSS == 2)
                {
                  d_pilotsTmp[0] = d_pilots2[0] * PILOT_P[d_pilotP];
                  d_pilotsTmp[1] = d_pilots2[1] * PILOT_P[d_pilotP];
                  d_pilotsTmp[2] = d_pilots2[2] * PILOT_P[d_pilotP];
                  d_pilotsTmp[3] = d_pilots2[3] * PILOT_P[d_pilotP];

                  procChipsToQam(&inBits2[i1], tmpSig1, d_m.mod, 52);
                  procInsertPilotsDc(tmpSig1, tmpSig2, d_pilotsTmp, d_m.format);
                  procNonDataSc(tmpSig2, tmpSig1, d_m.format);
                  procCSD(tmpSig1, -400);
                  ifft(tmpSig1, tmpSig2);
                  procToneScaling(tmpSig2, 56, d_m.nSS, 64);
                  memcpy(d_sigPtr2, &tmpSig2[48], sizeof(gr_complex) * 16);
                  d_sigPtr2 += 16;
                  memcpy(d_sigPtr2, tmpSig2, sizeof(gr_complex) * 64);
                  d_sigPtr2 += 64;
                  dout<<"ieee80211 modulation, gen non-legacy data sym ss 2: "<<d_nSymRd<<", total: "<<d_m.nSym<<std::endl;
                }
                pilotShift(d_pilots1);
                pilotShift(d_pilots2);
                d_pilotP = (d_pilotP + 1) % 127;
              }

              i1 += d_m.nSD;
              d_nSymRd++;
            }
            
            if(d_nSymRd >= d_m.nSym)
            {
              // tail pad
              memset(d_sigPtr1, 0, 1000 * sizeof(gr_complex));
              memset(d_sigPtr2, 0, 1000 * sizeof(gr_complex));
              d_sigPtr1 = d_sig1;
              d_sigPtr2 = d_sig2;
              if((d_m.nSym * d_m.nSD) < d_nChipsTotal)
              {
                d_sModul = MODUL_S_CLEAN;
              }
              else
              {
                d_sModul = MODUL_S_COPY;
              }
              
              if(d_m.format == C8P_F_L)
              {
                // total num of sym to be output
                d_nSampWrTotal = 1000 + (5 + d_m.nSym)*80 + 1000;   // begining gap, packet, end gap
                d_nSampWr = 0;
              }
              else if(d_m.format == C8P_F_VHT)
              {
                d_nSampWrTotal = 1000 + (5 + 4 + d_m.nLTF + d_m.nSym)*80 + 1000;
                d_nSampWr = 0;
              }
              else
              {
                d_nSampWrTotal = 1000 + (5 + 3 + d_m.nLTF + d_m.nSym)*80 + 1000;
                d_nSampWr = 0;
              }
              break;
            }
          }
          consume_each(i1);
          return 0;
        }

        case MODUL_S_CLEAN:
        {
          if(d_nProc >= (d_nChipsTotal - (d_m.nSym * d_m.nSD)))
          {
            dout<<"ieee80211 modulation, clean padded chips: "<<(d_nChipsTotal - (d_m.nSym * d_m.nSD))<<std::endl;
            consume_each(d_nChipsTotal - (d_m.nSym * d_m.nSD));
            d_sModul = MODUL_S_COPY;
          }
          else
          {
            consume_each(0);
          }
          return 0;
        }

        case MODUL_S_COPY:
        {
          int o = 0;
          
          if(d_nGen <= (d_nSampWrTotal - d_nSampWr))
          {
            o = d_nSampWrTotal - d_nSampWr;
            d_sModul = MODUL_S_IDLE;
          }
          else
          {
            o = d_nGen;
          }

          if(d_m.nSS == 1)
          {
            memcpy(outSig1, d_sigPtr1, o * sizeof(gr_complex));
            memset(outSig2, 0, o * sizeof(gr_complex));
          }
          else
          {
            memcpy(outSig1, d_sigPtr1, o * sizeof(gr_complex));
            memcpy(outSig2, d_sigPtr2, o * sizeof(gr_complex));
          }

          d_nSampWr += o;
          d_sigPtr1 += o;
          d_sigPtr2 += o;

          consume_each(0);
          return o;
        }

      }


      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (noutput_items);

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

    void
    modulation_impl::ifft(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_ifft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_ifft.execute();
      memcpy(res, d_ofdm_ifft.get_outbuf(), sizeof(gr_complex)*64);
    }

    void
    modulation_impl::pilotShift(gr_complex* pilots)
    {
      gr_complex tmpP = pilots[0];
      pilots[0] = pilots[1];
      pilots[1] = pilots[2];
      pilots[2] = pilots[3];
      pilots[3] = tmpP;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
