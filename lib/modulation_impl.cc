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

      // prepare training fields
      gr_complex tmpSig[64];
      // legacy stf and non legacy stf
      ifft(C8P_STF_F, tmpSig);
      memcpy(&d_stf_l[0], &tmpSig[32], 32);
      memcpy(&d_stf_l[32], &tmpSig[0], 64);
      memcpy(&d_stf_l[96], &tmpSig[0], 64);
      memcpy(&d_stf_nl[0], &tmpSig[48], 16);
      memcpy(&d_stf_nl[16], &tmpSig[0], 64);
      // legacy stf and non legacy stf with csd for 2nd stream
      memcpy(tmpSig, C8P_STF_F);
      procCSD(tmpSig, -200);
      ifft(tmpSig, &d_stf_l2[96]);
      memcpy(&d_stf_l2[0], &d_stf_l2[96], 32);
      memcpy(&d_stf_l2[32], &d_stf_l2[96], 64);
      memcpy(tmpSig, C8P_STF_F);
      procCSD(tmpSig, -400);
      ifft(tmpSig, &d_stf_nl2[16]);
      memcpy(&d_stf_nl2[0], &d_stf_l2[64], 16);
      // legacy ltf
      ifft(C8P_LTF_L_F, tmpSig);
      memcpy(&d_ltf_l[0], &tmpSig[32], 32);
      memcpy(&d_ltf_l[32], &tmpSig[0], 64);
      memcpy(&d_ltf_l[96], &tmpSig[0], 64);
      // legaycy ltf with csd
      memcpy(tmpSig, C8P_LTF_L_F);
      procCSD(tmpSig, -200);
      ifft(tmpSig, &d_ltf_l2[16]);
      memcpy(&d_ltf_l2[0], &d_ltf_l2[64], 16);
      // non legacy ltf
      ifft(C8P_LTF_NL_F, tmpSig);
      memcpy(&d_ltf_nl[0], &tmpSig[48], 16);
      memcpy(&d_ltf_nl[16], &tmpSig[0], 64);
      // non legaycy ltf with csd
      memcpy(tmpSig, C8P_LTF_NL_F);
      procCSD(tmpSig, -400);
      ifft(tmpSig, &d_ltf_nl2[16]);
      memcpy(&d_ltf_nl2[0], &d_ltf_nl2[64], 16);
      // non legacy ltf negative
      ifft(C8P_LTF_NL_F_N, tmpSig);
      memcpy(&d_ltf_nl_n[0], &tmpSig[48], 16);
      memcpy(&d_ltf_nl_n[16], &tmpSig[0], 64);
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
          int tmpTagFormat, tmpTagMcs, tmpTagNss, tmpTagLen;
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          // basic
          tmpTagFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(9999)));
          tmpTagMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(9999)));
          tmpTagNss = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss"), pmt::from_long(9999)));
          tmpTagLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(9999)));
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
          
          d_sModul = MODUL_S_MOD;
          consume_each(0);
          return 0;
        }

        case MODUL_S_MOD:
        {

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

  } /* namespace ieee80211 */
} /* namespace gr */
