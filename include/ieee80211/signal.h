/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Header
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

#ifndef INCLUDED_IEEE80211_SIGNAL_H
#define INCLUDED_IEEE80211_SIGNAL_H

#include <ieee80211/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace ieee80211 {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211
     *
     */
    class IEEE80211_API signal : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<signal> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211::signal.
       *
       * To avoid accidental use of raw pointers, ieee80211::signal's
       * constructor is in a private implementation
       * class. ieee80211::signal::make is the public interface for
       * creating new instances.
       */
      static sptr make(int nss);
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_SIGNAL_H */
