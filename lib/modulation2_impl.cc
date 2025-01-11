/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
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
#include "modulation2_impl.h"

namespace gr {
  namespace ieee80211 {

    modulation2::sptr
    modulation2::make()
    {
      return gnuradio::make_block_sptr<modulation2_impl>(
        );
    }


    /*
     * The private constructor
     */
    modulation2_impl::modulation2_impl()
      : gr::block("modulation2",
              gr::io_signature::make(2, 2, sizeof(uint8_t)),
              gr::io_signature::make(2, 2, sizeof(gr_complex)))
    {
      d_sModul = MODUL_S_RD_TAG;
      d_debug = false;
      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&modulation2_impl::msgRead, this, _1));
      // prepare training fields
      gr_complex tmpSig[64];
      memset((uint8_t*)d_sigl, 0, sizeof(gr_complex) * 64);
      memset((uint8_t*)d_signl, 0, sizeof(gr_complex) * 384);
      memset((uint8_t*)d_signl0, 0, sizeof(gr_complex) * 448);
      memset((uint8_t*)d_signl1, 0, sizeof(gr_complex) * 448);
      memset((uint8_t*)d_signl1vht, 0, sizeof(gr_complex) * 448);
      memset((uint8_t*)d_signl0mu, 0, sizeof(gr_complex) * 448);
      memset((uint8_t*)d_signl1mu, 0, sizeof(gr_complex) * 448);
      // non legacy stf
      memcpy(d_signl+192, C8P_STF_F, sizeof(gr_complex) * 64);
      memcpy(d_signl0+192, C8P_STF_F, sizeof(gr_complex) * 64);
      memcpy(tmpSig, C8P_STF_F, sizeof(gr_complex) * 64);
      procCSD(tmpSig, -200);
      memcpy(d_signl1+192, tmpSig, sizeof(gr_complex) * 64);
      memcpy(d_signl1vht+192, tmpSig, sizeof(gr_complex) * 64);
      // non legacy ltf
      memcpy(d_signl+256, C8P_LTF_NL_F, sizeof(gr_complex) * 64);
      memcpy(d_signl0+256, C8P_LTF_NL_F, sizeof(gr_complex) * 64);
      memcpy(d_signl0+320, C8P_LTF_NL_F_N, sizeof(gr_complex) * 64);
      memcpy(tmpSig, C8P_LTF_NL_F, sizeof(gr_complex) * 64);
      procCSD(tmpSig, -400);
      memcpy(d_signl1+256, tmpSig, sizeof(gr_complex) * 64);
      memcpy(d_signl1+320, tmpSig, sizeof(gr_complex) * 64);
      memcpy(d_signl1vht+256, tmpSig, sizeof(gr_complex) * 64);
      memcpy(tmpSig, C8P_LTF_NL_F_VHT22, sizeof(gr_complex) * 64);
      procCSD(tmpSig, -400);
      memcpy(d_signl1vht+320, tmpSig, sizeof(gr_complex) * 64);
      gr_complex tmpPilotL[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
      gr_complex tmpPilotNL[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
      gr_complex tmpPilotHT20[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
      gr_complex tmpPilotHT21[4] = {gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(-1.0f, 0.0f), gr_complex(1.0f, 0.0f)};
      for(int i=0;i<1408;i++)
      {
        for(int j=0;j<4;j++)
        {
          d_pilotsL[i][j] = tmpPilotL[j] * PILOT_P[(i+1)%127];
          d_pilotsHT[i][j] = tmpPilotNL[j] * PILOT_P[(i+3)%127];
          d_pilotsVHT[i][j] = tmpPilotNL[j] * PILOT_P[(i+4)%127];
          d_pilotsHT20[i][j] = tmpPilotHT20[j] * PILOT_P[(i+3)%127];
          d_pilotsHT21[i][j] = tmpPilotHT21[j] * PILOT_P[(i+3)%127];
        }
        gr_complex tmpPilot;
        tmpPilot = tmpPilotNL[0];
        tmpPilotNL[0] = tmpPilotNL[1];
        tmpPilotNL[1] = tmpPilotNL[2];
        tmpPilotNL[2] = tmpPilotNL[3];
        tmpPilotNL[3] = tmpPilot;
        tmpPilot = tmpPilotHT20[0];
        tmpPilotHT20[0] = tmpPilotHT20[1];
        tmpPilotHT20[1] = tmpPilotHT20[2];
        tmpPilotHT20[2] = tmpPilotHT20[3];
        tmpPilotHT20[3] = tmpPilot;
        tmpPilot = tmpPilotHT21[0];
        tmpPilotHT21[0] = tmpPilotHT21[1];
        tmpPilotHT21[1] = tmpPilotHT21[2];
        tmpPilotHT21[2] = tmpPilotHT21[3];
        tmpPilotHT21[3] = tmpPilot;
      }
    }

    void
    modulation2_impl::msgRead(pmt::pmt_t msg)
    {
      /* 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP*/
      pmt::pmt_t msgVec = pmt::cdr(msg);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(msgVec, tmpOffset);
      int tmpPktLen = pmt::blob_length(msgVec);
      int tmpPktFormat = tmpPkt[0];
      if(tmpPktFormat == C8P_F_VHT_BFQ && tmpPktLen==2049)
      {
        std::cout<<"ieee80211 mod, get bfq"<<std::endl;
        memcpy((uint8_t*) d_vhtMuBfQ, (tmpPkt + 1), sizeof(gr_complex) * 256);
        memcpy(d_signl0mu+192, d_signl0+192, sizeof(gr_complex) * 192);
        memcpy(d_signl1mu+192, d_signl1vht+192, sizeof(gr_complex) * 192);
        procNss2SymBfQ(d_signl0mu+192, d_signl1mu+192, d_vhtMuBfQ);
        procNss2SymBfQ(d_signl0mu+256, d_signl1mu+256, d_vhtMuBfQ);
        procNss2SymBfQ(d_signl0mu+320, d_signl1mu+320, d_vhtMuBfQ);
      }
    }

    /*
     * Our virtual destructor.
     */
    modulation2_impl::~modulation2_impl()
    {
    }

    void
    modulation2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
    }

    int
    modulation2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* inChips0 = static_cast<const uint8_t*>(input_items[0]);
      const uint8_t* inChips1 = static_cast<const uint8_t*>(input_items[1]);
      gr_complex* outSig0 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[1]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;

      if(d_sModul == MODUL_S_RD_TAG)
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
          d_sigBitsIntedL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigl"), pmt::PMT_NIL));
          gr_complex tmpSigPilots[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
          if(d_pktFormat == C8P_F_VHT_MU)
          {
            d_pktMcs1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs1"), pmt::from_long(-1)));
            d_pktNss1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss1"), pmt::from_long(-1)));
            d_pktLen1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len1"), pmt::from_long(-1)));
            std::cout<<"ieee80211 mod2, mu #"<<d_pktSeq<<", mcs0:"<<d_pktMcs0<<", nss0:"<<d_pktNss0<<", len0:"<<d_pktLen0<<", mcs1:"<<d_pktMcs1<<", nss1:"<<d_pktNss1<<", len1:"<<d_pktLen1<<std::endl;
            formatToModMu(&d_m, d_pktMcs0, 1, d_pktLen0, d_pktMcs1, 1, d_pktLen1);
            d_nSymCopied = 0;
            d_nSampSigCopied = 0;
            d_sigBitsIntedNL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("signl"), pmt::PMT_NIL));
            d_sigBitsIntedB0 = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigb0"), pmt::PMT_NIL));
            d_sigBitsIntedB1 = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigb1"), pmt::PMT_NIL));
            procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl0mu, C8P_QAM_BPSK);
            procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl0mu+64, C8P_QAM_BPSK);
            procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl0mu+128, C8P_QAM_QBPSK);
            procInsertPilots(d_signl0mu, tmpSigPilots);
            procInsertPilots(d_signl0mu+64, tmpSigPilots);
            procInsertPilots(d_signl0mu+128, tmpSigPilots);
            memcpy((uint8_t*)d_signl1mu, (uint8_t*)d_signl0mu, sizeof(gr_complex)*192);
            procChipsToQamNonShiftedScNL(&d_sigBitsIntedB0[0], d_signl0mu+384, C8P_QAM_BPSK);
            procChipsToQamNonShiftedScNL(&d_sigBitsIntedB1[0], d_signl1mu+384, C8P_QAM_BPSK);
            procInsertPilots(d_signl0mu+384, tmpSigPilots);//insert sigB0 pilot 
            procInsertPilots(d_signl1mu+384, tmpSigPilots);//insert sigB1 pilot 
            procCSD(d_signl1mu, -200);
            procCSD(d_signl1mu+64, -200);
            procCSD(d_signl1mu+128, -200);
            procCSD(d_signl1mu+384, -400);
            procNss2SymBfQ(d_signl0mu+384, d_signl1mu+384, d_vhtMuBfQ);
            d_sigP0 = d_signl0mu;
            d_sigP1 = d_signl1mu;
            d_nSampSigTotal = 448;
            dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(7+d_m.nSym+MODUL_N_PADSYM));
            dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(2));
          }
          else
          {
            std::cout<<"ieee80211 mod2, su #"<<d_pktSeq<<", format:"<<d_pktFormat<<", mcs:"<<d_pktMcs0<<", nss:"<<d_pktNss0<<", len:"<<d_pktLen0<<std::endl;
            formatToModSu(&d_m, d_pktFormat, d_pktMcs0, d_pktNss0, d_pktLen0);
            d_nSymCopied = 0;
            d_nSampSigCopied = 0;
            if(d_pktFormat == C8P_F_VHT)
            {
              d_sigBitsIntedNL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("signl"), pmt::PMT_NIL));
              d_sigBitsIntedB0 = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigb0"), pmt::PMT_NIL));
              if(d_m.nSS == 2)
              {
                procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl0, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl0+64, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl0+128, C8P_QAM_QBPSK);
                procChipsToQamNonShiftedScNL(&d_sigBitsIntedB0[0], d_signl0+384, C8P_QAM_BPSK);
                procInsertPilots(d_signl0, tmpSigPilots);
                procInsertPilots(d_signl0+64, tmpSigPilots);
                procInsertPilots(d_signl0+128, tmpSigPilots);
                procInsertPilots(d_signl0+384, tmpSigPilots);
                memcpy((uint8_t*)d_signl1vht, (uint8_t*)d_signl0, sizeof(gr_complex)*192);
                memcpy((uint8_t*)(d_signl1vht+384), (uint8_t*)(d_signl0+384), sizeof(gr_complex)*64);
                procCSD(d_signl1vht, -200);
                procCSD(d_signl1vht+64, -200);
                procCSD(d_signl1vht+128, -200);
                procCSD(d_signl1vht+384, -400);
                d_sigP0 = d_signl0;
                d_sigP1 = d_signl1vht;
                d_nSampSigTotal = 448;
                dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(7+d_m.nSym+MODUL_N_PADSYM));
              }
              else
              {
                procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl+64, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl+128, C8P_QAM_QBPSK);
                procChipsToQamNonShiftedScNL(&d_sigBitsIntedB0[0], d_signl+320, C8P_QAM_BPSK);
                procInsertPilots(d_signl, tmpSigPilots);
                procInsertPilots(d_signl+64, tmpSigPilots);
                procInsertPilots(d_signl+128, tmpSigPilots);
                procInsertPilots(d_signl+320, tmpSigPilots);
                d_sigP0 = d_signl;
                d_nSampSigTotal = 384;
                dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(6+d_m.nSym+MODUL_N_PADSYM));
              }
            }
            else if(d_pktFormat == C8P_F_HT)
            {
              d_sigBitsIntedNL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("signl"), pmt::PMT_NIL));
              if(d_m.nSS == 2)
              {
                procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl0, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl0+64, C8P_QAM_QBPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl0+128, C8P_QAM_QBPSK);
                procInsertPilots(d_signl0, tmpSigPilots);
                procInsertPilots(d_signl0+64, tmpSigPilots);
                procInsertPilots(d_signl0+128, tmpSigPilots);
                memcpy((uint8_t*)d_signl1, (uint8_t*)d_signl0, sizeof(gr_complex)*192);
                procCSD(d_signl1, -200);
                procCSD(d_signl1+64, -200);
                procCSD(d_signl1+128, -200);
                d_sigP0 = d_signl0;
                d_sigP1 = d_signl1;
                d_nSampSigTotal = 384;
                dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(6+d_m.nSym+MODUL_N_PADSYM));
              }
              else
              {
                procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl, C8P_QAM_BPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl+64, C8P_QAM_QBPSK);
                procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl+128, C8P_QAM_QBPSK);
                procInsertPilots(d_signl, tmpSigPilots);
                procInsertPilots(d_signl+64, tmpSigPilots);
                procInsertPilots(d_signl+128, tmpSigPilots);
                d_sigP0 = d_signl;
                d_nSampSigTotal = 320;
                dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(5+d_m.nSym+MODUL_N_PADSYM));
              }
            }
            else
            {
              procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_sigl, C8P_QAM_BPSK);
              procInsertPilots(d_sigl, tmpSigPilots);
              d_sigP0 = d_sigl;
              d_nSampSigTotal = 64;
              dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(1+d_m.nSym+MODUL_N_PADSYM));
            }
            dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_pktNss0));
          }
          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (size_t i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0,                   // output port index
                            nitems_written(0),  // output sample index
                            pmt::car(pair),     
                            pmt::cdr(pair),
                            alias_pmt());
              add_item_tag(1,                   // output port index
                            nitems_written(1),  // output sample index
                            pmt::car(pair),     
                            pmt::cdr(pair),
                            alias_pmt());
          }
          d_sModul = MODUL_S_SIG;
        }
      }

      if(d_sModul == MODUL_S_SIG)
      {
        if(d_nGen < (d_nSampSigTotal - d_nSampSigCopied))
        {
          memcpy(outSig0, d_sigP0 + d_nSampSigCopied, d_nGen * sizeof(gr_complex));
          if(d_m.nSS == 2)
          {
            memcpy(outSig1, d_sigP1 + d_nSampSigCopied, d_nGen * sizeof(gr_complex));
          }
          d_nGened += d_nGen;
          d_nSampSigCopied += d_nGen;
        }
        else
        {
          memcpy(outSig0, d_sigP0 + d_nSampSigCopied, (d_nSampSigTotal - d_nSampSigCopied) * sizeof(gr_complex));
          if(d_m.nSS == 2)
          {
            memcpy(outSig1, d_sigP1 + d_nSampSigCopied, (d_nSampSigTotal - d_nSampSigCopied) * sizeof(gr_complex));
          }
          d_nGened += (d_nSampSigTotal - d_nSampSigCopied);
          d_sModul = MODUL_S_DATA;
        }
      }

      if(d_sModul == MODUL_S_DATA)
      {
        while(true)
        {
          if(d_nSymCopied < (d_m.nSym+MODUL_N_PADSYM))
          {
            if(d_nSymCopied >= d_m.nSym && ((d_nGen - d_nGened) >= 64))
            {
              memset((uint8_t*)(outSig0 + d_nGened), 0, sizeof(gr_complex) * 64);
              memset((uint8_t*)(outSig1 + d_nGened), 0, sizeof(gr_complex) * 64);
              d_nSymCopied++;
              d_nGened+=64;
            }
            else if((d_nGen - d_nGened) >= 64 && (d_nProc - d_nProced) >= d_m.nSD)
            {
              if(d_m.sumu)
              {
                procChipsToQamNonShiftedScNL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.modMu[0]);
                procChipsToQamNonShiftedScNL(inChips1 + d_nProced, outSig1 + d_nGened, d_m.modMu[1]);
                procInsertPilots(outSig0 + d_nGened, d_pilotsVHT[d_nSymCopied]);
                procInsertPilots(outSig1 + d_nGened, d_pilotsVHT[d_nSymCopied]);
                procCSD(outSig1 + d_nGened, -400);
                procNss2SymBfQ(outSig0 + d_nGened, outSig1 + d_nGened, d_vhtMuBfQ);
              }
              else if(d_m.format == C8P_F_L)
              {
                procChipsToQamNonShiftedScL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                procInsertPilots(outSig0 + d_nGened, d_pilotsL[d_nSymCopied]);
                memset((uint8_t*)(outSig0 + d_nGened), 0, sizeof(gr_complex) * 6);
                memset((uint8_t*)(outSig0 + d_nGened + 59), 0, sizeof(gr_complex) * 5);
              }
              else if(d_m.format == C8P_F_VHT)
              {
                procChipsToQamNonShiftedScNL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                procInsertPilots(outSig0 + d_nGened, d_pilotsVHT[d_nSymCopied]);
                if(d_m.nSS == 2)
                {
                  procChipsToQamNonShiftedScNL(inChips1 + d_nProced, outSig1 + d_nGened, d_m.mod);
                  procInsertPilots(outSig1 + d_nGened, d_pilotsVHT[d_nSymCopied]);
                  procCSD(outSig1 + d_nGened, -400);
                }
              }
              else
              {
                procChipsToQamNonShiftedScNL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                if(d_m.nSS == 2)
                {
                  procChipsToQamNonShiftedScNL(inChips1 + d_nProced, outSig1 + d_nGened, d_m.mod);
                  procInsertPilots(outSig0 + d_nGened, d_pilotsHT20[d_nSymCopied]);
                  procInsertPilots(outSig1 + d_nGened, d_pilotsHT21[d_nSymCopied]);
                  procCSD(outSig1 + d_nGened, -400);
                }
                else
                {
                  procInsertPilots(outSig0 + d_nGened, d_pilotsHT[d_nSymCopied]);
                }
              }
              d_nSymCopied++;
              d_nProced+=d_m.nSD;
              d_nGened+=64;
            }
            else
            {
              break;
            }
          }
          else
          {
            d_sModul = MODUL_S_CLEAN;
            break;
          }
        }
      }

      if(d_sModul == MODUL_S_CLEAN)
      {
        if((d_nProc - d_nProced) >= MODUL_GR_GAP)
        {
          d_nProced += MODUL_GR_GAP;
          d_sModul = MODUL_S_RD_TAG;
        }
      }

      consume_each (d_nProced);
      return d_nGened;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
