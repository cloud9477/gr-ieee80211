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
      d_debug = false;

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
      // ninput_items_required[0] = noutput_items + 160;
      // ninput_items_required[1] = noutput_items + 160;
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
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
            d_sModul = MODUL_S_RD_TAG;
          }
          consume_each(0);
          return 0;
        }
        
        case MODUL_S_RD_TAG:
        {
          int tmpTagFormat, tmpTagMcs, tmpTagNss, tmpTagLen, tmpTagSeq;
          int tmpTagMcs1, tmpTagLen1;
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          // basic
          tmpTagFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(999999)));
          tmpTagSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(999999)));
          d_nChipsTotal = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("total"), pmt::from_long(999999)));
          if(tmpTagFormat == C8P_F_VHT_MU)
          {
            tmpTagMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs0"), pmt::from_long(999999)));
            tmpTagLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len0"), pmt::from_long(999999)));
            tmpTagMcs1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs1"), pmt::from_long(999999)));
            tmpTagLen1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len1"), pmt::from_long(999999)));
            formatToModMu(&d_m, tmpTagMcs, 1, tmpTagLen, tmpTagMcs1, 1, tmpTagLen1);
          }
          else
          {
            tmpTagMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(999999)));
            tmpTagNss = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss"), pmt::from_long(999999)));
            tmpTagLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(999999)));
            formatToModSu(&d_m, tmpTagFormat, tmpTagMcs, tmpTagNss, tmpTagLen);
          }
          // sig part
          d_tagLegacyBits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("lsig"), pmt::PMT_NIL));
          std::copy(d_tagLegacyBits.begin(), d_tagLegacyBits.end(), d_legacySigInted);
          if(d_m.format == C8P_F_VHT)
          {
            d_tagVhtABits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtsiga"), pmt::PMT_NIL));
            std::copy(d_tagVhtABits.begin(), d_tagVhtABits.end(), d_vhtSigAInted);
            d_tagVhtBBits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtsigb"), pmt::PMT_NIL));
            std::copy(d_tagVhtBBits.begin(), d_tagVhtBBits.end(), d_vhtSigBInted);
            if(d_m.sumu)
            {
              d_tagVhtBMu1Bits = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtsigb1"), pmt::PMT_NIL));
              std::copy(d_tagVhtBMu1Bits.begin(), d_tagVhtBMu1Bits.end(), d_vhtSigBMu1Inted);
              d_tagBfQ = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("vhtbfq"), pmt::PMT_NIL));
              std::copy(d_tagBfQ.begin(), d_tagBfQ.end(), d_vhtMuBfQ);
            }
          }
          else if(d_m.format == C8P_F_HT)
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

          // sig gap at begining
          memset((uint8_t*)d_sigPtr1, 0, sizeof(gr_complex) * MODUL_N_GAP);
          d_sigPtr1 += MODUL_N_GAP;
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

          if(d_m.format == C8P_F_L)
          {
            // set pilots for legacy
            d_pilots1[0] = gr_complex(1.0f, 0.0f);
            d_pilots1[1] = gr_complex(1.0f, 0.0f);
            d_pilots1[2] = gr_complex(1.0f, 0.0f);
            d_pilots1[3] = gr_complex(-1.0f, 0.0f);
            d_pilotP = 1;
            // tail pad
            memset((uint8_t*)d_sigPtr1, 0, MODUL_N_GAP * sizeof(gr_complex));
            consume_each(0);
            return 0;
          }
          else if(d_m.format == C8P_F_VHT)
          {
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
          if(d_m.sumu)
          {}
          else
          {
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
              procChipsToQam(d_vhtSigBInted, tmpSig1, C8P_QAM_BPSK, 52);
              procInsertPilotsDc(tmpSig1, tmpSig2, tmpSigPilots, C8P_F_VHT);
              procNonDataSc(tmpSig2, tmpSig1, C8P_F_VHT);
              ifft(tmpSig1, tmpSig2);
              procToneScaling(tmpSig2, 56, d_m.nSS, 64);
              memcpy(d_sigPtr1, &tmpSig2[48], sizeof(gr_complex) * 16);
              d_sigPtr1 += 16;
              memcpy(d_sigPtr1, tmpSig2, sizeof(gr_complex) * 64);
              d_sigPtr1 += 64;
            }
          }

          // ss 2 part
          if(d_m.nSS == 2)
          {
            // sig gap at begining
            memset((uint8_t*)d_sigPtr2, 0, sizeof(gr_complex) * MODUL_N_GAP);
            d_sigPtr2 += MODUL_N_GAP;
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

            if(d_m.format == C8P_F_VHT)
            {
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
            if(d_m.sumu)
            {
              gr_complex tmpSig3[64];
              // mu-mimo vht STF
              memcpy(tmpSig1, C8P_STF_F, 64*sizeof(gr_complex));
              memcpy(tmpSig2, C8P_STF_F, 64*sizeof(gr_complex));
              procCSD(tmpSig2, -400);
              procNss2SymBfQ(tmpSig1, tmpSig2, d_vhtMuBfQ);
              ifft(tmpSig1, tmpSig3);
              procToneScaling(tmpSig3, 12, d_m.nSS, 64);
              memcpy(d_sigPtr1, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr1 += 16;
              memcpy(d_sigPtr1, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr1 += 64;
              ifft(tmpSig2, tmpSig3);
              procToneScaling(tmpSig3, 12, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
              // mu-mimo vht LTF n=1
              memcpy(tmpSig1, C8P_LTF_NL_F, 64*sizeof(gr_complex));
              memcpy(tmpSig2, C8P_LTF_NL_F, 64*sizeof(gr_complex));
              procCSD(tmpSig2, -400);
              procNss2SymBfQ(tmpSig1, tmpSig2, d_vhtMuBfQ);
              ifft(tmpSig1, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr1, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr1 += 16;
              memcpy(d_sigPtr1, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr1 += 64;
              ifft(tmpSig2, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
              // mu-mimo vht LTF n=2
              memcpy(tmpSig1, C8P_LTF_NL_F_N, 64*sizeof(gr_complex));
              memcpy(tmpSig2, C8P_LTF_NL_F, 64*sizeof(gr_complex));
              procCSD(tmpSig2, -400);
              procNss2SymBfQ(tmpSig1, tmpSig2, d_vhtMuBfQ);
              ifft(tmpSig1, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr1, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr1 += 16;
              memcpy(d_sigPtr1, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr1 += 64;
              ifft(tmpSig2, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
              // mu-mimo vht sig b
              procChipsToQam(d_vhtSigBInted, tmpSig1, C8P_QAM_BPSK, 52);
              procInsertPilotsDc(tmpSig1, tmpSig3, tmpSigPilots, C8P_F_VHT);
              procNonDataSc(tmpSig3, tmpSig1, C8P_F_VHT);
              procChipsToQam(d_vhtSigBMu1Inted, tmpSig2, C8P_QAM_BPSK, 52);
              procInsertPilotsDc(tmpSig2, tmpSig3, tmpSigPilots, C8P_F_VHT);
              procNonDataSc(tmpSig3, tmpSig2, C8P_F_VHT);
              procCSD(tmpSig2, -400);
              procNss2SymBfQ(tmpSig1, tmpSig2, d_vhtMuBfQ);
              ifft(tmpSig1, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr1, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr1 += 16;
              memcpy(d_sigPtr1, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr1 += 64;
              ifft(tmpSig2, tmpSig3);
              procToneScaling(tmpSig3, 56, d_m.nSS, 64);
              memcpy(d_sigPtr2, &tmpSig3[48], sizeof(gr_complex) * 16);
              d_sigPtr2 += 16;
              memcpy(d_sigPtr2, tmpSig3, sizeof(gr_complex) * 64);
              d_sigPtr2 += 64;
            }
            else
            {
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
                procChipsToQam(d_vhtSigBInted, tmpSig1, C8P_QAM_BPSK, 52);
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
          }

          consume_each(0);
          return 0;
        }

        case MODUL_S_DATA:
        {
          gr_complex tmpSig1[64];
          gr_complex tmpSig2[64];
          gr_complex tmpSig3[64];
          int i1 = 0;
          while(((i1 + d_m.nSD) <= d_nProc))
          {
            if(d_m.nSym > 0)
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
              }
              else if(d_m.sumu)
              {
                d_pilotsTmp[0] = d_pilots1[0] * PILOT_P[d_pilotP];
                d_pilotsTmp[1] = d_pilots1[1] * PILOT_P[d_pilotP];
                d_pilotsTmp[2] = d_pilots1[2] * PILOT_P[d_pilotP];
                d_pilotsTmp[3] = d_pilots1[3] * PILOT_P[d_pilotP];
                procChipsToQam(&inBits1[i1], tmpSig1, d_m.mod, 52);
                procInsertPilotsDc(tmpSig1, tmpSig3, d_pilotsTmp, d_m.format);
                procNonDataSc(tmpSig3, tmpSig1, d_m.format);
                
                d_pilotsTmp[0] = d_pilots2[0] * PILOT_P[d_pilotP];
                d_pilotsTmp[1] = d_pilots2[1] * PILOT_P[d_pilotP];
                d_pilotsTmp[2] = d_pilots2[2] * PILOT_P[d_pilotP];
                d_pilotsTmp[3] = d_pilots2[3] * PILOT_P[d_pilotP];
                procChipsToQam(&inBits2[i1], tmpSig2, d_m.mod, 52);
                procInsertPilotsDc(tmpSig2, tmpSig3, d_pilotsTmp, d_m.format);
                procNonDataSc(tmpSig3, tmpSig2, d_m.format);
                procCSD(tmpSig2, -400);

                procNss2SymBfQ(tmpSig1, tmpSig2, d_vhtMuBfQ);

                ifft(tmpSig1, tmpSig3);
                procToneScaling(tmpSig3, 56, d_m.nSS, 64);
                memcpy(d_sigPtr1, &tmpSig3[48], sizeof(gr_complex) * 16);
                d_sigPtr1 += 16;
                memcpy(d_sigPtr1, tmpSig3, sizeof(gr_complex) * 64);
                d_sigPtr1 += 64;

                ifft(tmpSig2, tmpSig3);
                procToneScaling(tmpSig3, 56, d_m.nSS, 64);
                memcpy(d_sigPtr2, &tmpSig3[48], sizeof(gr_complex) * 16);
                d_sigPtr2 += 16;
                memcpy(d_sigPtr2, tmpSig3, sizeof(gr_complex) * 64);
                d_sigPtr2 += 64;

                pilotShift(d_pilots1);
                pilotShift(d_pilots2);
                d_pilotP = (d_pilotP + 1) % 127;

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
              memset((uint8_t*)d_sigPtr1, 0, MODUL_N_GAP * sizeof(gr_complex));
              memset((uint8_t*)d_sigPtr2, 0, MODUL_N_GAP * sizeof(gr_complex));
              d_sigPtr1 = d_sig1;
              d_sigPtr2 = d_sig2;
              d_sModul = MODUL_S_COPY;
              
              if(d_m.format == C8P_F_L)
              {
                // total num of sym to be output
                d_nSampWrTotal = MODUL_N_GAP + (5 + d_m.nSym)*80 + MODUL_N_GAP;   // begining gap, packet, end gap
                d_nSampWr = 0;
              }
              else if(d_m.format == C8P_F_VHT)
              {
                d_nSampWrTotal = MODUL_N_GAP + (5 + 4 + d_m.nLTF + d_m.nSym)*80 + MODUL_N_GAP;
                d_nSampWr = 0;
              }
              else
              {
                d_nSampWrTotal = MODUL_N_GAP + (5 + 3 + d_m.nLTF + d_m.nSym)*80 + MODUL_N_GAP;
                d_nSampWr = 0;
              }

              pmt::pmt_t dict = pmt::make_dict();
              dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(d_nSampWrTotal));
              pmt::pmt_t pairs = pmt::dict_items(dict);
              for (size_t i = 0; i < pmt::length(pairs); i++) {
                  pmt::pmt_t pair = pmt::nth(i, pairs);
                  add_item_tag(0,                   // output port index
                                nitems_written(0),  // output sample index
                                pmt::car(pair),
                                pmt::cdr(pair),
                                alias_pmt());
              }

              break;
            }
          }
          consume_each(i1);
          return 0;
        }

        case MODUL_S_COPY:
        {
          int o = 0;
          if(d_nGen >= (d_nSampWrTotal - d_nSampWr))
          {
            o = d_nSampWrTotal - d_nSampWr;
            d_sModul = MODUL_S_CLEAN;
          }
          else
          {
            o = d_nGen;
          }

          if(d_m.nSS == 1)
          {
            memcpy(outSig1, d_sigPtr1, o * sizeof(gr_complex));
            memset((uint8_t*)outSig2, 0, o * sizeof(gr_complex));
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

        case MODUL_S_CLEAN:
        {
          if(d_nProc >= (d_nChipsTotal - (d_m.nSym * d_m.nSD)))
          {
            consume_each(d_nChipsTotal - (d_m.nSym * d_m.nSD));
            d_sModul = MODUL_S_IDLE;
          }
          else
          {
            consume_each(0);
          }
          return 0;
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
      memcpy(&d_ifftShifted[0], &sig[32], sizeof(gr_complex)*32);
      memcpy(&d_ifftShifted[32], &sig[0], sizeof(gr_complex)*32);
      memcpy(d_ofdm_ifft.get_inbuf(), d_ifftShifted, sizeof(gr_complex)*64);
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
